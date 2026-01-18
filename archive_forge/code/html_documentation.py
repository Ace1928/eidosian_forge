import re
import six
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT
Remove potentially dangerous property declarations from CSS code.
        
        In particular, properties using the CSS ``url()`` function with a scheme
        that is not considered safe are removed:
        
        >>> sanitizer = HTMLSanitizer()
        >>> sanitizer.sanitize_css(u'''
        ...   background: url(javascript:alert("foo"));
        ...   color: #000;
        ... ''')
        ['color: #000']
        
        Also, the proprietary Internet Explorer function ``expression()`` is
        always stripped:
        
        >>> sanitizer.sanitize_css(u'''
        ...   background: #fff;
        ...   color: #000;
        ...   width: e/**/xpression(alert("foo"));
        ... ''')
        ['background: #fff', 'color: #000']
        
        :param text: the CSS text; this is expected to be `unicode` and to not
                     contain any character or numeric references
        :return: a list of declarations that are considered safe
        :rtype: `list`
        :since: version 0.4.3
        