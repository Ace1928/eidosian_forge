import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def _parse_comments(self):
    """Return a dictionary containing tab file comments (PRIVATE)."""
    comments = {}
    while True:
        if 'BLAST' in self.line and 'processed' not in self.line:
            program_line = self.line[len(' #'):].split(' ')
            comments['program'] = program_line[0].lower()
            comments['version'] = program_line[1]
        elif 'Query' in self.line:
            query_line = self.line[len('# Query: '):].split(' ', 1)
            comments['id'] = query_line[0]
            if len(query_line) == 2:
                comments['description'] = query_line[1]
        elif 'Database' in self.line:
            comments['target'] = self.line[len('# Database: '):]
        elif 'RID' in self.line:
            comments['rid'] = self.line[len('# RID: '):]
        elif 'Fields' in self.line:
            comments['fields'] = self._parse_fields_line()
        elif ' hits found' in self.line or 'processed' in self.line:
            self.line = self.handle.readline().strip()
            return comments
        self.line = self.handle.readline()
        if not self.line:
            return comments
        else:
            self.line = self.line.strip()