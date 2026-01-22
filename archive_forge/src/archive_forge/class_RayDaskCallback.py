import contextlib
from collections import namedtuple, defaultdict
from datetime import datetime
from dask.callbacks import Callback
class RayDaskCallback(Callback):
    """
    Extends Dask's `Callback` class with Ray-specific hooks. When instantiating
    or subclassing this class, both the normal Dask hooks (e.g. pretask,
    posttask, etc.) and the Ray-specific hooks can be provided.

    See `dask.callbacks.Callback` for usage.

    Caveats: Any Dask-Ray scheduler must bring the Ray-specific callbacks into
    context using the `local_ray_callbacks` context manager, since the built-in
    `local_callbacks` context manager provided by Dask isn't aware of this
    class.
    """
    ray_active = set()

    def __init__(self, **kwargs):
        """
        Ray-specific callbacks:
            - def _ray_presubmit(task, key, deps):
                Run before submitting a Ray task. If this callback returns a
                non-`None` value, a Ray task will _not_ be created and this
                value will be used as the would-be task's result value.

                Args:
                    task: A Dask task, where the first tuple item is
                        the task function, and the remaining tuple items are
                        the task arguments (either the actual argument values,
                        or Dask keys into the deps dictionary whose
                        corresponding values are the argument values).
                    key: The Dask graph key for the given task.
                    deps: The dependencies of this task.

                Returns:
                    Either None, in which case a Ray task will be submitted, or
                    a non-None value, in which case a Ray task will not be
                    submitted and this return value will be used as the
                    would-be task result value.

            - def _ray_postsubmit(task, key, deps, object_ref):
                Run after submitting a Ray task.

                Args:
                    task: A Dask task, where the first tuple item is
                        the task function, and the remaining tuple items are
                        the task arguments (either the actual argument values,
                        or Dask keys into the deps dictionary whose
                        corresponding values are the argument values).
                    key: The Dask graph key for the given task.
                    deps: The dependencies of this task.
                    object_ref (ray.ObjectRef): The object reference for the
                        return value of the Ray task.

            - def _ray_pretask(key, object_refs):
                Run before executing a Dask task within a Ray task. This
                executes after the task has been submitted, within a Ray
                worker. The return value of this task will be passed to the
                _ray_posttask callback, if provided.

                Args:
                    key: The Dask graph key for the Dask task.
                    object_refs (List[ray.ObjectRef]): The object references
                        for the arguments of the Ray task.

                Returns:
                    A value that will be passed to the corresponding
                    _ray_posttask callback, if said callback is defined.

            - def _ray_posttask(key, result, pre_state):
                Run after executing a Dask task within a Ray task. This
                executes within a Ray worker. This callback receives the return
                value of the _ray_pretask callback, if provided.

                Args:
                    key: The Dask graph key for the Dask task.
                    result: The task result value.
                    pre_state: The return value of the corresponding
                        _ray_pretask callback, if said callback is defined.

            - def _ray_postsubmit_all(object_refs, dsk):
                Run after all Ray tasks have been submitted.

                Args:
                    object_refs (List[ray.ObjectRef]): The object references
                        for the output (leaf) Ray tasks of the task graph.
                    dsk: The Dask graph.

            - def _ray_finish(result):
                Run after all Ray tasks have finished executing and the final
                result has been returned.

                Args:
                    result: The final result (output) of the Dask
                        computation, before any repackaging is done by
                        Dask collection-specific post-compute callbacks.
        """
        for cb in CBS:
            cb_func = kwargs.pop(cb, None)
            if cb_func is not None:
                setattr(self, '_' + cb, cb_func)
        super().__init__(**kwargs)

    @property
    def _ray_callback(self):
        return RayCallback(*[getattr(self, field, None) for field in CB_FIELDS])

    def __enter__(self):
        self._ray_cm = add_ray_callbacks(self)
        self._ray_cm.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self._ray_cm.__exit__(*args)

    def register(self):
        type(self).ray_active.add(self._ray_callback)
        super().register()

    def unregister(self):
        type(self).ray_active.remove(self._ray_callback)
        super().unregister()