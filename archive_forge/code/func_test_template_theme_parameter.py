from bokeh.document import Document
from panel.template.fast.list import FastListTemplate
from panel.theme.fast import FastDarkTheme
def test_template_theme_parameter():
    template = FastListTemplate(title='Fast', theme='dark')
    doc = template.server_doc(Document())
    assert doc.theme._json['attrs']['figure']['background_fill_color'] == '#181818'
    assert isinstance(template._design.theme, FastDarkTheme)