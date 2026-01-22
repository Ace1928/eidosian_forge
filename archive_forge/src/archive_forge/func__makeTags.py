import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _makeTags(tagStr, xml):
    """Internal helper to construct opening and closing tag expressions, given a tag name"""
    if isinstance(tagStr, basestring):
        resname = tagStr
        tagStr = Keyword(tagStr, caseless=not xml)
    else:
        resname = tagStr.name
    tagAttrName = Word(alphas, alphanums + '_-:')
    if xml:
        tagAttrValue = dblQuotedString.copy().setParseAction(removeQuotes)
        openTag = Suppress('<') + tagStr('tag') + Dict(ZeroOrMore(Group(tagAttrName + Suppress('=') + tagAttrValue))) + Optional('/', default=[False]).setResultsName('empty').setParseAction(lambda s, l, t: t[0] == '/') + Suppress('>')
    else:
        printablesLessRAbrack = ''.join((c for c in printables if c not in '>'))
        tagAttrValue = quotedString.copy().setParseAction(removeQuotes) | Word(printablesLessRAbrack)
        openTag = Suppress('<') + tagStr('tag') + Dict(ZeroOrMore(Group(tagAttrName.setParseAction(downcaseTokens) + Optional(Suppress('=') + tagAttrValue)))) + Optional('/', default=[False]).setResultsName('empty').setParseAction(lambda s, l, t: t[0] == '/') + Suppress('>')
    closeTag = Combine(_L('</') + tagStr + '>')
    openTag = openTag.setResultsName('start' + ''.join(resname.replace(':', ' ').title().split())).setName('<%s>' % tagStr)
    closeTag = closeTag.setResultsName('end' + ''.join(resname.replace(':', ' ').title().split())).setName('</%s>' % tagStr)
    openTag.tag = resname
    closeTag.tag = resname
    return (openTag, closeTag)