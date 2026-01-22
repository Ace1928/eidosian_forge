import textwrap
def _master_name(self, ix):
    if self.merger is not None:
        ttf = self.merger.ttfs[ix]
        if 'name' in ttf and ttf['name'].getBestFullName():
            return ttf['name'].getBestFullName()
        elif hasattr(ttf.reader, 'file') and hasattr(ttf.reader.file, 'name'):
            return ttf.reader.file.name
    return f'master number {ix}'