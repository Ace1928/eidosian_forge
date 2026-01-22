import re
def _strip_xml_ns(tag):
    return tag.split('}', 1)[1] if '}' in tag else tag