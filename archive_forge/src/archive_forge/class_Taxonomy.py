import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
class Taxonomy(PhyloElement):
    """Describe taxonomic information for a clade.

    :Parameters:
        id_source : Id
            link other elements to a taxonomy (on the XML level)
        id : Id
            unique identifier of a taxon, e.g. Id('6500',
            provider='ncbi_taxonomy') for the California sea hare
        code : restricted string
            store UniProt/Swiss-Prot style organism codes, e.g. 'APLCA' for the
            California sea hare 'Aplysia californica'
        scientific_name : string
            the standard scientific name for this organism, e.g. 'Aplysia
            californica' for the California sea hare
        authority : string
            keep the authority, such as 'J. G. Cooper, 1863', associated with
            the 'scientific_name'
        common_names : list of strings
            common names for this organism
        synonyms : list of strings
            synonyms for this taxon?
        rank : restricted string
            taxonomic rank
        uri : Uri
            link
        other : list of Other objects
            non-phyloXML elements

    """
    re_code = re.compile('[a-zA-Z0-9_]{2,10}')
    ok_rank = {'domain', 'kingdom', 'subkingdom', 'branch', 'infrakingdom', 'superphylum', 'phylum', 'subphylum', 'infraphylum', 'microphylum', 'superdivision', 'division', 'subdivision', 'infradivision', 'superclass', 'class', 'subclass', 'infraclass', 'superlegion', 'legion', 'sublegion', 'infralegion', 'supercohort', 'cohort', 'subcohort', 'infracohort', 'superorder', 'order', 'suborder', 'superfamily', 'family', 'subfamily', 'supertribe', 'tribe', 'subtribe', 'infratribe', 'genus', 'subgenus', 'superspecies', 'species', 'subspecies', 'variety', 'subvariety', 'form', 'subform', 'cultivar', 'unknown', 'other'}

    def __init__(self, id_source=None, id=None, code=None, scientific_name=None, authority=None, rank=None, uri=None, common_names=None, synonyms=None, other=None):
        """Initialize the class."""
        _check_str(code, self.re_code.match)
        _check_str(rank, self.ok_rank.__contains__)
        self.id_source = id_source
        self.id = id
        self.code = code
        self.scientific_name = scientific_name
        self.authority = authority
        self.rank = rank
        self.uri = uri
        self.common_names = common_names or []
        self.synonyms = synonyms or []
        self.other = other or []

    def __str__(self):
        """Show the class name and an identifying attribute."""
        if self.code is not None:
            return self.code
        if self.scientific_name is not None:
            return self.scientific_name
        if self.rank is not None:
            return self.rank
        if self.id is not None:
            return str(self.id)
        return self.__class__.__name__