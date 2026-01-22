import copy
import glob
import itertools
import os
import uuid
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import warnings
import numpy as np
from ray.air._internal.usage import tag_searcher
from ray.tune.error import TuneError
from ray.tune.experiment.config_parser import _make_parser, _create_trial_from_spec
from ray.tune.search.sample import np_random_generator, _BackwardsCompatibleNumpyRng
from ray.tune.search.variant_generator import (
from ray.tune.search.search_algorithm import SearchAlgorithm
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
from ray.util import PublicAPI
@PublicAPI
class BasicVariantGenerator(SearchAlgorithm):
    """Uses Tune's variant generation for resolving variables.

    This is the default search algorithm used if no other search algorithm
    is specified.


    Args:
        points_to_evaluate: Initial parameter suggestions to be run
            first. This is for when you already have some good parameters
            you want to run first to help the algorithm make better suggestions
            for future parameters. Needs to be a list of dicts containing the
            configurations.
        max_concurrent: Maximum number of concurrently running trials.
            If 0 (default), no maximum is enforced.
        constant_grid_search: If this is set to ``True``, Ray Tune will
            *first* try to sample random values and keep them constant over
            grid search parameters. If this is set to ``False`` (default),
            Ray Tune will sample new random parameters in each grid search
            condition.
        random_state:
            Seed or numpy random generator to use for reproducible results.
            If None (default), will use the global numpy random generator
            (``np.random``). Please note that full reproducibility cannot
            be guaranteed in a distributed environment.


    Example:

    .. code-block:: python

        from ray import tune

        # This will automatically use the `BasicVariantGenerator`
        tuner = tune.Tuner(
            lambda config: config["a"] + config["b"],
            tune_config=tune.TuneConfig(
                num_samples=4
            ),
            param_space={
                "a": tune.grid_search([1, 2]),
                "b": tune.randint(0, 3)
            },
        )
        tuner.fit()

    In the example above, 8 trials will be generated: For each sample
    (``4``), each of the grid search variants for ``a`` will be sampled
    once. The ``b`` parameter will be sampled randomly.

    The generator accepts a pre-set list of points that should be evaluated.
    The points will replace the first samples of each experiment passed to
    the ``BasicVariantGenerator``.

    Each point will replace one sample of the specified ``num_samples``. If
    grid search variables are overwritten with the values specified in the
    presets, the number of samples will thus be reduced.

    Example:

    .. code-block:: python

        from ray import tune
        from ray.tune.search.basic_variant import BasicVariantGenerator

        tuner = tune.Tuner(
            lambda config: config["a"] + config["b"],
            tune_config=tune.TuneConfig(
                search_alg=BasicVariantGenerator(points_to_evaluate=[
                    {"a": 2, "b": 2},
                    {"a": 1},
                    {"b": 2}
                ]),
                num_samples=4
            ),
            param_space={
                "a": tune.grid_search([1, 2]),
                "b": tune.randint(0, 3)
            },
        )
        tuner.fit()

    The example above will produce six trials via four samples:

    - The first sample will produce one trial with ``a=2`` and ``b=2``.
    - The second sample will produce one trial with ``a=1`` and ``b`` sampled
      randomly
    - The third sample will produce two trials, one for each grid search
      value of ``a``. It will be ``b=2`` for both of these trials.
    - The fourth sample will produce two trials, one for each grid search
      value of ``a``. ``b`` will be sampled randomly and independently for
      both of these trials.

    """
    CKPT_FILE_TMPL = 'basic-variant-state-{}.json'

    def __init__(self, points_to_evaluate: Optional[List[Dict]]=None, max_concurrent: int=0, constant_grid_search: bool=False, random_state: Optional[Union[int, 'np_random_generator', np.random.RandomState]]=None):
        tag_searcher(self)
        self._trial_generator = []
        self._iterators = []
        self._trial_iter = None
        self._finished = False
        self._random_state = _BackwardsCompatibleNumpyRng(random_state)
        self._points_to_evaluate = points_to_evaluate or []
        force_test_uuid = os.environ.get('_TEST_TUNE_TRIAL_UUID')
        if force_test_uuid:
            self._uuid_prefix = force_test_uuid + '_'
        else:
            self._uuid_prefix = str(uuid.uuid1().hex)[:5] + '_'
        self._total_samples = 0
        self.max_concurrent = max_concurrent
        self._constant_grid_search = constant_grid_search
        self._live_trials = set()

    @property
    def total_samples(self):
        return self._total_samples

    def add_configurations(self, experiments: Union['Experiment', List['Experiment'], Dict[str, Dict]]):
        """Chains generator given experiment specifications.

        Arguments:
            experiments: Experiments to run.
        """
        from ray.tune.experiment import _convert_to_experiment_list
        experiment_list = _convert_to_experiment_list(experiments)
        for experiment in experiment_list:
            grid_vals = _count_spec_samples(experiment.spec, num_samples=1)
            lazy_eval = grid_vals > SERIALIZATION_THRESHOLD
            if lazy_eval:
                warnings.warn(f'The number of pre-generated samples ({grid_vals}) exceeds the serialization threshold ({int(SERIALIZATION_THRESHOLD)}). Resume ability is disabled. To fix this, reduce the number of dimensions/size of the provided grid search.')
            previous_samples = self._total_samples
            points_to_evaluate = copy.deepcopy(self._points_to_evaluate)
            self._total_samples += _count_variants(experiment.spec, points_to_evaluate)
            iterator = _TrialIterator(uuid_prefix=self._uuid_prefix, num_samples=experiment.spec.get('num_samples', 1), unresolved_spec=experiment.spec, constant_grid_search=self._constant_grid_search, points_to_evaluate=points_to_evaluate, lazy_eval=lazy_eval, start=previous_samples, random_state=self._random_state)
            self._iterators.append(iterator)
            self._trial_generator = itertools.chain(self._trial_generator, iterator)

    def next_trial(self):
        """Provides one Trial object to be queued into the TrialRunner.

        Returns:
            Trial: Returns a single trial.
        """
        if self.is_finished():
            return None
        if self.max_concurrent > 0 and len(self._live_trials) >= self.max_concurrent:
            return None
        if not self._trial_iter:
            self._trial_iter = iter(self._trial_generator)
        try:
            trial = next(self._trial_iter)
            self._live_trials.add(trial.trial_id)
            return trial
        except StopIteration:
            self._trial_generator = []
            self._trial_iter = None
            self.set_finished()
            return None

    def on_trial_complete(self, trial_id: str, result: Optional[Dict]=None, error: bool=False):
        if trial_id in self._live_trials:
            self._live_trials.remove(trial_id)

    def get_state(self):
        if any((iterator.lazy_eval for iterator in self._iterators)):
            return False
        state = self.__dict__.copy()
        del state['_trial_generator']
        return state

    def set_state(self, state):
        self.__dict__.update(state)
        for iterator in self._iterators:
            self._trial_generator = itertools.chain(self._trial_generator, iterator)

    def save_to_dir(self, dirpath, session_str):
        if any((iterator.lazy_eval for iterator in self._iterators)):
            return False
        state_dict = self.get_state()
        _atomic_save(state=state_dict, checkpoint_dir=dirpath, file_name=self.CKPT_FILE_TMPL.format(session_str), tmp_file_name='.tmp_generator')

    def has_checkpoint(self, dirpath: str):
        """Whether a checkpoint file exists within dirpath."""
        return bool(glob.glob(os.path.join(dirpath, self.CKPT_FILE_TMPL.format('*'))))

    def restore_from_dir(self, dirpath: str):
        """Restores self + searcher + search wrappers from dirpath."""
        state_dict = _load_newest_checkpoint(dirpath, self.CKPT_FILE_TMPL.format('*'))
        if not state_dict:
            raise RuntimeError('Unable to find checkpoint in {}.'.format(dirpath))
        self.set_state(state_dict)