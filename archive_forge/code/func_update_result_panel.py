import pytest
import panel as pn
from panel.widgets import Grammar, GrammarList, SpeechToText
from panel.widgets.speech_to_text import (
@pn.depends(speech_to_text_color, watch=True)
def update_result_panel(results_last):
    results_last = results_last.lower()
    if results_last in colors:
        app.styles = dict(background=results_last)
        result_panel.object = 'Result received: ' + results_last
    else:
        app.styles = dict(background='white')
        result_panel.object = 'Result received: ' + results_last + ' (Not recognized)'