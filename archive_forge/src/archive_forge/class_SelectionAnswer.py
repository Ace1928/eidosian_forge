import xml.sax.saxutils
class SelectionAnswer(object):
    """
    A class to generate SelectionAnswer XML data structures.
    Does not yet implement Binary selection options.
    """
    SELECTIONANSWER_XML_TEMPLATE = '<SelectionAnswer>%s%s<Selections>%s</Selections></SelectionAnswer>'
    SELECTION_XML_TEMPLATE = '<Selection><SelectionIdentifier>%s</SelectionIdentifier>%s</Selection>'
    SELECTION_VALUE_XML_TEMPLATE = '<%s>%s</%s>'
    STYLE_XML_TEMPLATE = '<StyleSuggestion>%s</StyleSuggestion>'
    MIN_SELECTION_COUNT_XML_TEMPLATE = '<MinSelectionCount>%s</MinSelectionCount>'
    MAX_SELECTION_COUNT_XML_TEMPLATE = '<MaxSelectionCount>%s</MaxSelectionCount>'
    ACCEPTED_STYLES = ['radiobutton', 'dropdown', 'checkbox', 'list', 'combobox', 'multichooser']
    OTHER_SELECTION_ELEMENT_NAME = 'OtherSelection'

    def __init__(self, min=1, max=1, style=None, selections=None, type='text', other=False):
        if style is not None:
            if style in SelectionAnswer.ACCEPTED_STYLES:
                self.style_suggestion = style
            else:
                raise ValueError("style '%s' not recognized; should be one of %s" % (style, ', '.join(SelectionAnswer.ACCEPTED_STYLES)))
        else:
            self.style_suggestion = None
        if selections is None:
            raise ValueError('SelectionAnswer.__init__(): selections must be a non-empty list of (content, identifier) tuples')
        else:
            self.selections = selections
        self.min_selections = min
        self.max_selections = max
        assert len(selections) >= self.min_selections, '# of selections is less than minimum of %d' % self.min_selections
        self.type = type
        self.other = other

    def get_as_xml(self):
        if self.type == 'text':
            TYPE_TAG = 'Text'
        elif self.type == 'binary':
            TYPE_TAG = 'Binary'
        else:
            raise ValueError("illegal type: %s; must be either 'text' or 'binary'" % str(self.type))
        selections_xml = ''
        for tpl in self.selections:
            value_xml = SelectionAnswer.SELECTION_VALUE_XML_TEMPLATE % (TYPE_TAG, tpl[0], TYPE_TAG)
            selection_xml = SelectionAnswer.SELECTION_XML_TEMPLATE % (tpl[1], value_xml)
            selections_xml += selection_xml
        if self.other:
            if hasattr(self.other, 'get_as_xml'):
                assert isinstance(self.other, FreeTextAnswer), 'OtherSelection can only be a FreeTextAnswer'
                selections_xml += self.other.get_as_xml().replace('FreeTextAnswer', 'OtherSelection')
            else:
                selections_xml += '<OtherSelection />'
        if self.style_suggestion is not None:
            style_xml = SelectionAnswer.STYLE_XML_TEMPLATE % self.style_suggestion
        else:
            style_xml = ''
        if self.style_suggestion != 'radiobutton':
            count_xml = SelectionAnswer.MIN_SELECTION_COUNT_XML_TEMPLATE % self.min_selections
            count_xml += SelectionAnswer.MAX_SELECTION_COUNT_XML_TEMPLATE % self.max_selections
        else:
            count_xml = ''
        ret = SelectionAnswer.SELECTIONANSWER_XML_TEMPLATE % (count_xml, style_xml, selections_xml)
        return ret