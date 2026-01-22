import os
def ShellEscapeList(words):
    """Turn a list of words into a shell-safe string.

  Args:
    words: A list of words, e.g. for a command.

  Returns:
    A string of shell-quoted and space-separated words.
  """
    if win32:
        return ' '.join(words)
    s = ''
    for word in words:
        s += "'" + word.replace("'", '\'"\'"\'') + "' "
    return s[:-1]