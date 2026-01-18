from .agents import BASE_PYTHON_TOOLS, clean_code_for_chat
from .python_interpreter import InterpretorError, evaluate
def translator(text, src_lang, tgt_lang):
    return f'This is the translation of {text} from {src_lang} to {tgt_lang}.'