from reportlab.lib import colors
from ._Colors import ColorTranslator
def __process_feature(self):
    """Examine wrapped feature and set some properties accordingly (PRIVATE)."""
    self.locations = []
    bounds = []
    for location in self._feature.location.parts:
        start = int(location.start)
        end = int(location.end)
        self.locations.append((start, end))
        bounds += [start, end]
    self.type = str(self._feature.type)
    if self._feature.location.strand is None:
        self.strand = 0
    else:
        self.strand = int(self._feature.location.strand)
    if 'color' in self._feature.qualifiers:
        self.color = self._colortranslator.artemis_color(self._feature.qualifiers['color'][0])
    self.name = self.type
    for qualifier in self.name_qualifiers:
        if qualifier in self._feature.qualifiers:
            self.name = self._feature.qualifiers[qualifier][0]
            break
    self.start, self.end = (min(bounds), max(bounds))