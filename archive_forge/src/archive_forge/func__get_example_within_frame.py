import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _get_example_within_frame(self, vnframe):
    """Returns example within a frame

        A utility function to retrieve an example within a frame in VerbNet.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        :return: example_text: The example sentence for this particular frame
        """
    example_element = vnframe.find('EXAMPLES/EXAMPLE')
    if example_element is not None:
        example_text = example_element.text
    else:
        example_text = ''
    return example_text