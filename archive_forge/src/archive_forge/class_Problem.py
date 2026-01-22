from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
class Problem:
    """
    A class regrouping all the information to solve a problem on which we will evaluate agents.

    Args:
        task (`str` ou `list[str]`):
            One or several descriptions of the task to perform. If a list, it should contain variations on the
            phrasing, but for the same task.
        inputs (`list[str]` or `dict[str, str]`):
            The inputs that will be fed to the tools. For this testing environment, only strings are accepted as
            values. Pass along a dictionary when you want to specify the values of each inputs, or just the list of
            inputs expected (the value used will be `<<input_name>>` in this case).
        answer (`str` or `list[str`]):
            The theoretical answer (or list of possible valid answers) to the problem, as code.
    """

    def __init__(self, task, inputs, answer):
        self.task = task
        self.inputs = inputs
        self.answer = answer