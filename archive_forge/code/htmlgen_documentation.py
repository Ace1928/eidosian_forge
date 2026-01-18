import xml.etree.ElementTree as ET
from html import escape

Kind of like htmlgen, only much simpler.  The only important symbol
that is exported is ``html``.

This builds ElementTree nodes, but with some extra useful methods.
(Open issue: should it use ``ElementTree`` more, and the raw
``Element`` stuff less?)

You create tags with attribute access.  I.e., the ``A`` anchor tag is
``html.a``.  The attributes of the HTML tag are done with keyword
arguments.  The contents of the tag are the non-keyword arguments
(concatenated).  You can also use the special ``c`` keyword, passing a
list, tuple, or single tag, and it will make up the contents (this is
useful because keywords have to come after all non-keyword arguments,
which is non-intuitive).  Or you can chain them, adding the keywords
with one call, then the body with a second call, like::

    >>> str(html.a(href='http://yahoo.com')('<Yahoo>'))
    '<a href="http://yahoo.com">&lt;Yahoo&gt;</a>'

Note that strings will be quoted; only tags given explicitly will
remain unquoted.

If the value of an attribute is None, then no attribute
will be inserted.  So::

    >>> str(html.a(href='http://www.yahoo.com', name=None,
    ...              c='Click Here'))
    '<a href="http://www.yahoo.com">Click Here</a>'

If the value is None, then the empty string is used.  Otherwise str()
is called on the value.

``html`` can also be called, and it will produce a special list from
its arguments, which adds a ``__str__`` method that does ``html.str``
(which handles quoting, flattening these lists recursively, and using
'' for ``None``).

``html.comment`` will generate an HTML comment, like
``html.comment('comment text')`` -- note that it cannot take keyword
arguments (because they wouldn't mean anything).

Examples::

    >>> str(html.html(
    ...    html.head(html.title("Page Title")),
    ...    html.body(
    ...    bgcolor='#000066',
    ...    text='#ffffff',
    ...    c=[html.h1('Page Title'),
    ...       html.p('Hello world!')],
    ...    )))
    '<html><head><title>Page Title</title></head><body bgcolor="#000066" text="#ffffff"><h1>Page Title</h1><p>Hello world!</p></body></html>'
    >>> str(html.a(href='#top')('return to top'))
    '<a href="#top">return to top</a>'

