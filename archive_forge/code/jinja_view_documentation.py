import copy
import jinja2
Set the Template Text

        Sets the text of the current template, marking it
        for recompilation next time the compiled template
        is retrieved via attr:`template` .

        :param str textval: the new text of the Jinja template
        