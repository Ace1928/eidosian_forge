import re
import lxml
import lxml.etree
from lxml.html.clean import Cleaner
def traverse_text_fragments(tree, context, handle_tail=True):
    """ Extract text from the ``tree``: fill ``chunks`` variable """
    add_newlines(tree.tag, context)
    add_text(tree.text, context)
    for child in tree:
        traverse_text_fragments(child, context)
    add_newlines(tree.tag, context)
    if handle_tail:
        add_text(tree.tail, context)