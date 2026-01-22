import argparse
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
def _analyze_tab_complete_input(self, text):
    """Analyze raw input to tab-completer.

    Args:
      text: (str) the full, raw input text to be tab-completed.

    Returns:
      context: (str) the context str. For example,
        If text == "print_tensor softmax", returns "print_tensor".
        If text == "print", returns "".
        If text == "", returns "".
      prefix: (str) the prefix to be tab-completed, from the last word.
        For example, if text == "print_tensor softmax", returns "softmax".
        If text == "print", returns "print".
        If text == "", returns "".
      except_last_word: (str) the input text, except the last word.
        For example, if text == "print_tensor softmax", returns "print_tensor".
        If text == "print_tensor -a softmax", returns "print_tensor -a".
        If text == "print", returns "".
        If text == "", returns "".
    """
    text = text.lstrip()
    if not text:
        context = ''
        prefix = ''
        except_last_word = ''
    else:
        items = text.split(' ')
        if len(items) == 1:
            context = ''
            prefix = items[0]
            except_last_word = ''
        else:
            context = items[0]
            prefix = items[-1]
            except_last_word = ' '.join(items[:-1]) + ' '
    return (context, prefix, except_last_word)