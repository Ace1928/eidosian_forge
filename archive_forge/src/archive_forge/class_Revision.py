from typing import Dict, List, Optional, Tuple
from . import errors, osutils
class Revision:
    """Single revision on a branch.

    Revisions may know their revision_hash, but only once they've been
    written out.  This is not stored because you cannot write the hash
    into the file it describes.

    Attributes:
      parent_ids: List of parent revision_ids

      properties:
        Dictionary of revision properties.  These are attached to the
        revision as extra metadata.  The name must be a single
        word; the value can be an arbitrary string.
    """
    parent_ids: List[RevisionID]
    revision_id: RevisionID
    parent_sha1s: List[str]
    committer: Optional[str]
    message: str
    properties: Dict[str, bytes]
    inventory_sha1: str
    timestamp: float
    timezone: int

    def __init__(self, revision_id: RevisionID, properties=None, **args) -> None:
        self.revision_id = revision_id
        if properties is None:
            self.properties = {}
        else:
            self.properties = properties
            self._check_properties()
        self.committer = None
        self.parent_ids = []
        self.parent_sha1s = []
        self.__dict__.update(args)

    def __repr__(self):
        return '<Revision id %s>' % self.revision_id

    def datetime(self):
        import datetime
        return datetime.datetime.fromtimestamp(self.timestamp)

    def __eq__(self, other):
        if not isinstance(other, Revision):
            return False
        return self.inventory_sha1 == other.inventory_sha1 and self.revision_id == other.revision_id and (self.timestamp == other.timestamp) and (self.message == other.message) and (self.timezone == other.timezone) and (self.committer == other.committer) and (self.properties == other.properties) and (self.parent_ids == other.parent_ids)

    def __ne__(self, other):
        return not self.__eq__(other)

    def _check_properties(self):
        """Verify that all revision properties are OK."""
        for name, value in self.properties.items():
            not_text = not isinstance(name, str)
            if not_text or osutils.contains_whitespace(name):
                raise ValueError('invalid property name %r' % name)
            if not isinstance(value, (str, bytes)):
                raise ValueError('invalid property value %r for %r' % (value, name))

    def get_history(self, repository):
        """Return the canonical line-of-history for this revision.

        If ghosts are present this may differ in result from a ghost-free
        repository.
        """
        current_revision = self
        reversed_result = []
        while current_revision is not None:
            reversed_result.append(current_revision.revision_id)
            if not len(current_revision.parent_ids):
                reversed_result.append(None)
                current_revision = None
            else:
                next_revision_id = current_revision.parent_ids[0]
                current_revision = repository.get_revision(next_revision_id)
        reversed_result.reverse()
        return reversed_result

    def get_summary(self):
        """Get the first line of the log message for this revision.

        Return an empty string if message is None.
        """
        if self.message:
            return self.message.lstrip().split('\n', 1)[0]
        else:
            return ''

    def get_apparent_authors(self):
        """Return the apparent authors of this revision.

        If the revision properties contain the names of the authors,
        return them. Otherwise return the committer name.

        The return value will be a list containing at least one element.
        """
        authors = self.properties.get('authors', None)
        if authors is None:
            author = self.properties.get('author', self.committer)
            if author is None:
                return []
            return [author]
        else:
            return authors.split('\n')

    def iter_bugs(self):
        """Iterate over the bugs associated with this revision."""
        bug_property = self.properties.get('bugs', None)
        if bug_property is None:
            return iter([])
        from . import bugtracker
        return bugtracker.decode_bug_urls(bug_property)