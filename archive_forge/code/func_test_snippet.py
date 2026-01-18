from io import StringIO
from html.parser import HTMLParser
import json
import os
import re
import tempfile
import shutil
import traitlets
from ..widgets import IntSlider, IntText, Text, Widget, jslink, HBox, widget_serialization, widget as widget_module
from ..embed import embed_data, embed_snippet, embed_minimal_html, dependency_state
def test_snippet(self):

    class Parser(HTMLParser):
        state = 'initial'
        states = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'script' and attrs.get('type', '') == 'application/vnd.jupyter.widget-state+json':
                self.state = 'widget-state'
                self.states.append(self.state)
            elif tag == 'script' and attrs.get('type', '') == 'application/vnd.jupyter.widget-view+json':
                self.state = 'widget-view'
                self.states.append(self.state)

        def handle_endtag(self, tag):
            self.state = 'initial'

        def handle_data(self, data):
            if self.state == 'widget-state':
                manager_state = json.loads(data)['state']
                assert len(manager_state) == 3
                self.states.append('check-widget-state')
            elif self.state == 'widget-view':
                view = json.loads(data)
                assert isinstance(view, dict)
                self.states.append('check-widget-view')
    w = IntText(4)
    state = dependency_state(w, drop_defaults=True)
    snippet = embed_snippet(views=w, drop_defaults=True, state=state)
    parser = Parser()
    parser.feed(snippet)
    print(parser.states)
    assert parser.states == ['widget-state', 'check-widget-state', 'widget-view', 'check-widget-view']