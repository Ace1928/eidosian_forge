from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def transcriber(audio):
    if 'sound' not in audio:
        raise ValueError(f'`audio` ({audio}) is not a sound.')
    return f'This is the transcribed text from {audio}.'