import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def _fixup_ins_del_tags(doc):
    """fixup_ins_del_tags that works on an lxml document in-place
    """
    for tag in ['ins', 'del']:
        for el in doc.xpath('descendant-or-self::%s' % tag):
            if not _contains_block_level_tag(el):
                continue
            _move_el_inside_block(el, tag=tag)
            el.drop_tag()