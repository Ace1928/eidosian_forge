from .exceptions import ParseException
from .util import col, replaced_by_pep8
def with_class(classname, namespace=''):
    """
    Simplified version of :class:`with_attribute` when
    matching on a div class - made difficult because ``class`` is
    a reserved word in Python.

    Example::

        html = '''
            <div>
            Some text
            <div class="grid">1 4 0 1 0</div>
            <div class="graph">1,3 2,3 1,1</div>
            <div>this &lt;div&gt; has no class</div>
            </div>

        '''
        div,div_end = make_html_tags("div")
        div_grid = div().set_parse_action(with_class("grid"))

        grid_expr = div_grid + SkipTo(div | div_end)("body")
        for grid_header in grid_expr.search_string(html):
            print(grid_header.body)

        div_any_type = div().set_parse_action(with_class(withAttribute.ANY_VALUE))
        div_expr = div_any_type + SkipTo(div | div_end)("body")
        for div_header in div_expr.search_string(html):
            print(div_header.body)

    prints::

        1 4 0 1 0

        1 4 0 1 0
        1,3 2,3 1,1
    """
    classattr = f'{namespace}:class' if namespace else 'class'
    return with_attribute(**{classattr: classname})