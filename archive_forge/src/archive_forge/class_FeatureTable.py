import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
class FeatureTable(SeqFeature):
    """Stores feature annotations for specific regions of the sequence.

    This is a subclass of SeqFeature, defined in Bio.SeqFeature, where the
    attributes are used as follows:

     - ``location``: location of the feature on the canonical or isoform
       sequence; the location is stored as an instance of SimpleLocation,
       defined in Bio.SeqFeature, with the ref attribute set to the isoform
       ID referring to the canonical or isoform sequence on which the feature
       is defined
     - ``id``: unique and stable identifier (FTId), only provided for features
       belonging to the types CARBOHYD, CHAIN, PEPTIDE, PROPEP, VARIANT, or
       VAR_SEQ
     - ``type``: indicates the type of feature, as defined by the UniProt
       Knowledgebase documentation:

        - ACT_SITE: amino acid(s) involved in the activity of an enzyme
        - BINDING:  binding site for any chemical group
        - CARBOHYD: glycosylation site; an FTId identifier to the GlyConnect
          database is provided if annotated there
        - CA_BIND:  calcium-binding region
        - CHAIN:    polypeptide chain in the mature protein
        - COILED:   coiled-coil region
        - COMPBIAS: compositionally biased region
        - CONFLICT: different sources report differing sequences
        - CROSSLNK: posttransationally formed amino acid bond
        - DISULFID: disulfide bond
        - DNA_BIND: DNA-binding region
        - DOMAIN:   domain, defined as a specific combination of secondary
          structures organized into a characteristic three-dimensional
          structure or fold
        - INIT_MET: initiator methionine
        - INTRAMEM: region located in a membrane without crossing it
        - HELIX:    alpha-, 3(10)-, or pi-helix secondary structure
        - LIPID:    covalent binding of a lipid moiety
        - METAL:    binding site for a metal ion
        - MOD_RES:  posttranslational modification (PTM) of a residue,
          annotated by the controlled vocabulary defined by the ptmlist.txt
          document on the UniProt website
        - MOTIF:    short sequence motif of biological interest
        - MUTAGEN:  site experimentally altered by mutagenesis
        - NON_CONS: non-consecutive residues
        - NON_STD:  non-standard amino acid
        - NON_TER:  the residue at an extremity of the sequence is not the
          terminal residue
        - NP_BIND:  nucleotide phosphate-binding region
        - PEPTIDE:  released active mature polypeptide
        - PROPEP:   any processed propeptide
        - REGION:   region of interest in the sequence
        - REPEAT:   internal sequence repetition
        - SIGNAL:   signal sequence (prepeptide)
        - SITE:     amino-acid site of interest not represented by another
          feature key
        - STRAND:   beta-strand secondary structure; either a hydrogen-bonded
          extended beta strand or a residue in an isolated beta-bridge
        - TOPO_DOM: topological domain
        - TRANSIT:  transit peptide (mitochondrion, chloroplast, thylakoid,
          cyanelle, peroxisome, etc.)
        - TRANSMEM: transmembrane region
        - TURN:     H-bonded turn (3-, 4-, or 5-turn)
        - UNSURE:   uncertainties in the sequence
        - VARIANT:  sequence variant; an FTId is provided for protein sequence
          variants of Hominidae (great apes and humans)
        - VAR_SEQ:  sequence variant produced by alternative splicing,
          alternative promoter usage, alternative initiation, or ribosomal
          frameshifting
        - ZN_FING:  zinc finger region

     - ``qualifiers``: a dictionary of additional information, which may include
       the feature evidence and free-text notes. While SwissProt includes the
       feature identifier code (FTId) as a qualifier, it is stored as the
       attribute ID of the FeatureTable object.

    """