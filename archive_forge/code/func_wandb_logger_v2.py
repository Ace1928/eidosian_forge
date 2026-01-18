from typing import Dict, Any, Tuple, Callable, List, IO, Optional
from types import ModuleType
import sys
from spacy import Language
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def wandb_logger_v2(project_name: str, remove_config_values: List[str]=SimpleFrozenList(), model_log_interval: Optional[int]=None, log_dataset_dir: Optional[str]=None) -> LoggerT:
    """Creates a logger that interoperates with the Weights & Biases framework.

    Args:
        project_name (str):
            The name of the project in the Weights & Biases interface. The project will be created automatically if it doesn't exist yet.
        remove_config_values (List[str]):
            A list of values to exclude from the config before it is uploaded to W&B. Defaults to [].
        model_log_interval (Optional[int]):
            Steps to wait between logging model checkpoints to the W&B dasboard. Defaults to None.
        log_dataset_dir (Optional[str]):
            Directory containing the dataset to be logged and versioned as a W&B artifact. Defaults to None.

    Returns:
        LoggerT: Logger instance.
    """
    wandb = _import_wandb()

    def setup_logger(nlp: 'Language', stdout: IO=sys.stdout, stderr: IO=sys.stderr) -> Tuple[Callable[[Dict[str, Any]], None], Callable[[], None]]:
        console_log_step, console_finalize = setup_default_console_logger(nlp, stdout, stderr)
        run = _setup_wandb(wandb, nlp, project_name, remove_config_values=remove_config_values)
        if log_dataset_dir:
            _log_dir_artifact(wandb, path=log_dataset_dir, name='dataset', type='dataset')

        def log_step(info: Optional[Dict[str, Any]]):
            console_log_step(info)
            _log_scores(wandb, info)
            _log_model_artifact(wandb, info, run, model_log_interval)

        def finalize() -> None:
            console_finalize()
            wandb.join()
        return (log_step, finalize)
    return setup_logger