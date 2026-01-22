import re
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
class _HitPlaceholder:

    def createHit(self, hsp_list):
        hit = Hit(hsp_list)
        hit.id_ = self.id_
        hit.evalue = self.evalue
        hit.bitscore = self.bitscore
        if self.description:
            hit.description = self.description
        hit.domain_obs_num = self.domain_obs_num
        return hit