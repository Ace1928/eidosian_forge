from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def video_generator(prompt, seconds=2):
    return f'A video of {prompt}'