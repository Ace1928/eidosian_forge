import ast
from nbconvert.preprocessors import Preprocessor
class OutputMagicProcessor(Preprocessor):
    """
    Preprocessor to convert notebooks to Python source to convert use of
    output magic to use the util.output utility instead.
    """

    def preprocess_cell(self, cell, resources, index):
        if cell['cell_type'] == 'code':
            source = replace_line_magic(cell['source'], '%output', template='hv.util.output({line!r})')
            source, output_lines = filter_magic(source, '%%output')
            if output_lines:
                template = f'hv.util.output({output_lines[-1]!r}, {{expr}})'
                source = wrap_cell_expression(source, template)
            cell['source'] = source
        return (cell, resources)

    def __call__(self, nb, resources):
        return self.preprocess(nb, resources)