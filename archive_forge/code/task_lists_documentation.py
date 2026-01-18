import re
A mistune plugin to support task lists. Spec defined by
    GitHub flavored Markdown and commonly used by many parsers:

    .. code-block:: text

        - [ ] unchecked task
        - [x] checked task

    :param md: Markdown instance
    