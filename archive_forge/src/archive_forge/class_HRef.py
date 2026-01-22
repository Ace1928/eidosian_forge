from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
class HRef(BaseMultipartText):
    """
    A :py:class:`HRef` represends a hyperlink:

    >>> from pybtex.richtext import Tag
    >>> href = HRef('http://ctan.org/', 'CTAN')
    >>> print(href.render_as('html'))
    <a href="http://ctan.org/">CTAN</a>
    >>> print(href.render_as('latex'))
    \\href{http://ctan.org/}{CTAN}

    >>> href = HRef(String('http://ctan.org/'), String('http://ctan.org/'))
    >>> print(href.render_as('latex'))
    \\url{http://ctan.org/}

    :py:class:`HRef` supports the same methods as :py:class:`Text`.

    """

    def __init__(self, url, *args):
        if not isinstance(url, (six.string_types, BaseText)):
            raise ValueError('url must be str or Text (got %s)' % url.__class__.__name__)
        self.url = six.text_type(url)
        self.info = (self.url,)
        super(HRef, self).__init__(*args)

    def __repr__(self):
        reprparts = ', '.join((repr(part) for part in self.parts))
        return 'HRef({}, {})'.format(str_repr(self.url), reprparts)

    def render(self, backend):
        text = super(HRef, self).render(backend)
        return backend.format_href(self.url, text)