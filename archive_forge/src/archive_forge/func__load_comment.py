from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _load_comment(self, record, bioentry_id):
    """Record a SeqRecord's annotated comment in the database (PRIVATE).

        Arguments:
         - record - a SeqRecord object with an annotated comment
         - bioentry_id - corresponding database identifier

        """
    comments = record.annotations.get('comment')
    if not comments:
        return
    if not isinstance(comments, list):
        comments = [comments]
    for index, comment in enumerate(comments):
        comment = comment.replace('\n', ' ')
        sql = 'INSERT INTO comment (bioentry_id, comment_text, "rank") VALUES (%s, %s, %s)'
        self.adaptor.execute(sql, (bioentry_id, comment, index + 1))