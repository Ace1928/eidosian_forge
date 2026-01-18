import logging
import re
Parse from the contents that make up a watch file.

        :param lines: watch file lines to parse
        :return: instance or None if there are no non-comment lines in the file
        :raise MissingVersion: if there is no version number declared
        :raise ValueError: when syntax errors are encountered
        