import xml.sax.saxutils
class QuestionForm(ValidatingXML, list):
    """
    From the AMT API docs:

    The top-most element of the QuestionForm data structure is a
    QuestionForm element. This element contains optional Overview
    elements and one or more Question elements. There can be any
    number of these two element types listed in any order. The
    following example structure has an Overview element and a
    Question element followed by a second Overview element and
    Question element--all within the same QuestionForm.

    ::

        <QuestionForm xmlns="[the QuestionForm schema URL]">
            <Overview>
                [...]
            </Overview>
            <Question>
                [...]
            </Question>
            <Overview>
                [...]
            </Overview>
            <Question>
                [...]
            </Question>
            [...]
        </QuestionForm>

    QuestionForm is implemented as a list, so to construct a
    QuestionForm, simply append Questions and Overviews (with at least
    one Question).
    """
    schema_url = 'http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionForm.xsd'
    xml_template = '<QuestionForm xmlns="%(schema_url)s">%%(items)s</QuestionForm>' % vars()

    def is_valid(self):
        return any((isinstance(item, Question) for item in self)) and all((isinstance(item, (Question, Overview)) for item in self))

    def get_as_xml(self):
        assert self.is_valid(), 'QuestionForm contains invalid elements'
        items = ''.join((item.get_as_xml() for item in self))
        return self.xml_template % vars()