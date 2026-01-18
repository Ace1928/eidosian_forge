from paste.wsgilib import catch_errors_app
from paste.response import has_header, header_value, replace_header
from paste.request import resolve_relative_url
from paste.util.quoting import strip_html, html_quote, no_quote, comment_quote
 text/html representation of the exception 