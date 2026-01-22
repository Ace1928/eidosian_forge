from ..common.utils import bytes2str
def _strip_type_tag(die):
    """Given a DIE with DW_TAG_foo_type, returns foo"""
    return die.tag[7:-5]