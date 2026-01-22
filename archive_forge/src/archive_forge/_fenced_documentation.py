import re
from ._base import DirectiveParser, BaseDirective
A **fenced** style of directive looks like a fenced code block, it is
    inspired by markdown-it-docutils. The syntax looks like:

    .. code-block:: text

        ```{directive-type} title
        :option-key: option value
        :option-key: option value

        content text here
        ```

    To use ``FencedDirective``, developers can add it into plugin list in
    the :class:`Markdown` instance:

    .. code-block:: python

        import mistune
        from mistune.directives import FencedDirective, Admonition

        md = mistune.create_markdown(plugins=[
            # ...
            FencedDirective([Admonition()]),
        ])

    FencedDirective is using >= 3 backticks or curly-brackets for the fenced
    syntax. Developers can change it to other characters, e.g. colon:

    .. code-block:: python

            directive = FencedDirective([Admonition()], ':')

    And then the directive syntax would look like:

    .. code-block:: text

        ::::{note} Nesting directives
        You can nest directives by ensuring the start and end fence matching
        the length. For instance, in this example, the admonition is started
        with 4 colons, then it should end with 4 colons.

        You can nest another admonition with other length of colons except 4.

        :::{tip} Longer outermost fence
        It would be better that you put longer markers for the outer fence,
        and shorter markers for the inner fence. In this example, we put 4
        colons outsie, and 3 colons inside.
        :::
        ::::

    :param plugins: list of directive plugins
    :param markers: characters to determine the fence, default is backtick
                    and curly-bracket
    