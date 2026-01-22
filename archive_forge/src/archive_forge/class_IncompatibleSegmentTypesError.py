class IncompatibleSegmentTypesError(IncompatibleGlyphsError):

    def __init__(self, glyphs, segments):
        IncompatibleGlyphsError.__init__(self, glyphs)
        self.segments = segments

    def __str__(self):
        lines = []
        ndigits = len(str(max(self.segments)))
        for i, tags in sorted(self.segments.items()):
            lines.append('%s: (%s)' % (str(i).rjust(ndigits), ', '.join((repr(t) for t in tags))))
        return 'Glyphs named %s have incompatible segment types:\n  %s' % (self.combined_name, '\n  '.join(lines))