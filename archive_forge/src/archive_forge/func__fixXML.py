import re
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
def _fixXML(text):
    """
    Fix the various issues with Senseval pseudo-XML.
    """
    text = re.sub('<([~\\^])>', '\\1', text)
    text = re.sub('(\\s+)\\&(\\s+)', '\\1&amp;\\2', text)
    text = re.sub('"""', '\'"\'', text)
    text = re.sub('(<[^<]*snum=)([^">]+)>', '\\1"\\2"/>', text)
    text = re.sub('<\\&frasl>\\s*<p[^>]*>', 'FRASL', text)
    text = re.sub('<\\&I[^>]*>', '', text)
    text = re.sub('<{([^}]+)}>', '\\1', text)
    text = re.sub('<(@|/?p)>', '', text)
    text = re.sub('<&\\w+ \\.>', '', text)
    text = re.sub('<!DOCTYPE[^>]*>', '', text)
    text = re.sub('<\\[\\/?[^>]+\\]*>', '', text)
    text = re.sub('<(\\&\\w+;)>', '\\1', text)
    text = re.sub('&(?!amp|gt|lt|apos|quot)', '', text)
    text = re.sub('[ \\t]*([^<>\\s]+?)[ \\t]*<p="([^"]*"?)"/>', ' <wf pos="\\2">\\1</wf>', text)
    text = re.sub('\\s*"\\s*<p=\\\'"\\\'/>', ' <wf pos=\'"\'>"</wf>', text)
    return text