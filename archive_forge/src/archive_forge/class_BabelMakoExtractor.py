from babel.messages.extract import extract_python
from mako.ext.extract import MessageExtractor
class BabelMakoExtractor(MessageExtractor):

    def __init__(self, keywords, comment_tags, options):
        self.keywords = keywords
        self.options = options
        self.config = {'comment-tags': ' '.join(comment_tags), 'encoding': options.get('input_encoding', options.get('encoding', None))}
        super().__init__()

    def __call__(self, fileobj):
        return self.process_file(fileobj)

    def process_python(self, code, code_lineno, translator_strings):
        comment_tags = self.config['comment-tags']
        for lineno, funcname, messages, python_translator_comments in extract_python(code, self.keywords, comment_tags, self.options):
            yield (code_lineno + (lineno - 1), funcname, messages, translator_strings + python_translator_comments)