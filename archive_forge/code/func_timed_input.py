import sys
import threading
import wandb
def timed_input(prompt: str, timeout: float, show_timeout: bool=True, jupyter: bool=False) -> str:
    """Behaves like builtin `input()` but adds timeout.

    Args:
        prompt (str): Prompt to output to stdout.
        timeout (float): Timeout to wait for input.
        show_timeout (bool): Show timeout in prompt
        jupyter (bool): If True, use jupyter specific code.

    Raises:
        TimeoutError: exception raised if timeout occurred.
    """
    if show_timeout:
        prompt = f'{prompt}({timeout:.0f} second timeout) '
    if jupyter:
        return _jupyter_timed_input(prompt=prompt, timeout=timeout)
    return _timed_input(prompt=prompt, timeout=timeout)