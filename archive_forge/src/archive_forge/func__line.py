import re
import textwrap
def _line(self, text, indent=0):
    """Write 'text' word-wrapped at self.width characters."""
    leading_space = '  ' * indent
    while len(leading_space) + len(text) > self.width:
        available_space = self.width - len(leading_space) - len(' $')
        space = available_space
        while True:
            space = text.rfind(' ', 0, space)
            if space < 0 or self._count_dollars_before_index(text, space) % 2 == 0:
                break
        if space < 0:
            space = available_space - 1
            while True:
                space = text.find(' ', space + 1)
                if space < 0 or self._count_dollars_before_index(text, space) % 2 == 0:
                    break
        if space < 0:
            break
        self.output.write(leading_space + text[0:space] + ' $\n')
        text = text[space + 1:]
        leading_space = '  ' * (indent + 2)
    self.output.write(leading_space + text + '\n')