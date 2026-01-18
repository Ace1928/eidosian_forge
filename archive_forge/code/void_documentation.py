import collections
from rdflib.graph import Graph
from rdflib.namespace import RDF, VOID
from rdflib.term import Literal, URIRef

    Returns a new graph with a VoID description of the passed dataset

    For more info on Vocabulary of Interlinked Datasets (VoID), see:
    http://vocab.deri.ie/void

    This only makes two passes through the triples (once to detect the types
    of things)

    The tradeoff is that lots of temporary structures are built up in memory
    meaning lots of memory may be consumed :)
    I imagine at least a few copies of your original graph.

    the distinctForPartitions parameter controls whether
    distinctSubjects/objects are tracked for each class/propertyPartition
    this requires more memory again

    