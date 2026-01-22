from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DependencyEdge(_messages.Message):
    """Represents dependency parse tree information for a token. (For more
  information on dependency labels, see
  http://www.aclweb.org/anthology/P13-2017

  Enums:
    LabelValueValuesEnum: The parse label for the token.

  Fields:
    headTokenIndex: Represents the head of this token in the dependency tree.
      This is the index of the token which has an arc going to this token. The
      index is the position of the token in the array of tokens returned by
      the API method. If this token is a root token, then the
      `head_token_index` is its own index.
    label: The parse label for the token.
  """

    class LabelValueValuesEnum(_messages.Enum):
        """The parse label for the token.

    Values:
      UNKNOWN: Unknown
      ABBREV: Abbreviation modifier
      ACOMP: Adjectival complement
      ADVCL: Adverbial clause modifier
      ADVMOD: Adverbial modifier
      AMOD: Adjectival modifier of an NP
      APPOS: Appositional modifier of an NP
      ATTR: Attribute dependent of a copular verb
      AUX: Auxiliary (non-main) verb
      AUXPASS: Passive auxiliary
      CC: Coordinating conjunction
      CCOMP: Clausal complement of a verb or adjective
      CONJ: Conjunct
      CSUBJ: Clausal subject
      CSUBJPASS: Clausal passive subject
      DEP: Dependency (unable to determine)
      DET: Determiner
      DISCOURSE: Discourse
      DOBJ: Direct object
      EXPL: Expletive
      GOESWITH: Goes with (part of a word in a text not well edited)
      IOBJ: Indirect object
      MARK: Marker (word introducing a subordinate clause)
      MWE: Multi-word expression
      MWV: Multi-word verbal expression
      NEG: Negation modifier
      NN: Noun compound modifier
      NPADVMOD: Noun phrase used as an adverbial modifier
      NSUBJ: Nominal subject
      NSUBJPASS: Passive nominal subject
      NUM: Numeric modifier of a noun
      NUMBER: Element of compound number
      P: Punctuation mark
      PARATAXIS: Parataxis relation
      PARTMOD: Participial modifier
      PCOMP: The complement of a preposition is a clause
      POBJ: Object of a preposition
      POSS: Possession modifier
      POSTNEG: Postverbal negative particle
      PRECOMP: Predicate complement
      PRECONJ: Preconjunt
      PREDET: Predeterminer
      PREF: Prefix
      PREP: Prepositional modifier
      PRONL: The relationship between a verb and verbal morpheme
      PRT: Particle
      PS: Associative or possessive marker
      QUANTMOD: Quantifier phrase modifier
      RCMOD: Relative clause modifier
      RCMODREL: Complementizer in relative clause
      RDROP: Ellipsis without a preceding predicate
      REF: Referent
      REMNANT: Remnant
      REPARANDUM: Reparandum
      ROOT: Root
      SNUM: Suffix specifying a unit of number
      SUFF: Suffix
      TMOD: Temporal modifier
      TOPIC: Topic marker
      VMOD: Clause headed by an infinite form of the verb that modifies a noun
      VOCATIVE: Vocative
      XCOMP: Open clausal complement
      SUFFIX: Name suffix
      TITLE: Name title
      ADVPHMOD: Adverbial phrase modifier
      AUXCAUS: Causative auxiliary
      AUXVV: Helper auxiliary
      DTMOD: Rentaishi (Prenominal modifier)
      FOREIGN: Foreign words
      KW: Keyword
      LIST: List for chains of comparable items
      NOMC: Nominalized clause
      NOMCSUBJ: Nominalized clausal subject
      NOMCSUBJPASS: Nominalized clausal passive
      NUMC: Compound of numeric modifier
      COP: Copula
      DISLOCATED: Dislocated relation (for fronted/topicalized elements)
      ASP: Aspect marker
      GMOD: Genitive modifier
      GOBJ: Genitive object
      INFMOD: Infinitival modifier
      MES: Measure
      NCOMP: Nominal complement of a noun
    """
        UNKNOWN = 0
        ABBREV = 1
        ACOMP = 2
        ADVCL = 3
        ADVMOD = 4
        AMOD = 5
        APPOS = 6
        ATTR = 7
        AUX = 8
        AUXPASS = 9
        CC = 10
        CCOMP = 11
        CONJ = 12
        CSUBJ = 13
        CSUBJPASS = 14
        DEP = 15
        DET = 16
        DISCOURSE = 17
        DOBJ = 18
        EXPL = 19
        GOESWITH = 20
        IOBJ = 21
        MARK = 22
        MWE = 23
        MWV = 24
        NEG = 25
        NN = 26
        NPADVMOD = 27
        NSUBJ = 28
        NSUBJPASS = 29
        NUM = 30
        NUMBER = 31
        P = 32
        PARATAXIS = 33
        PARTMOD = 34
        PCOMP = 35
        POBJ = 36
        POSS = 37
        POSTNEG = 38
        PRECOMP = 39
        PRECONJ = 40
        PREDET = 41
        PREF = 42
        PREP = 43
        PRONL = 44
        PRT = 45
        PS = 46
        QUANTMOD = 47
        RCMOD = 48
        RCMODREL = 49
        RDROP = 50
        REF = 51
        REMNANT = 52
        REPARANDUM = 53
        ROOT = 54
        SNUM = 55
        SUFF = 56
        TMOD = 57
        TOPIC = 58
        VMOD = 59
        VOCATIVE = 60
        XCOMP = 61
        SUFFIX = 62
        TITLE = 63
        ADVPHMOD = 64
        AUXCAUS = 65
        AUXVV = 66
        DTMOD = 67
        FOREIGN = 68
        KW = 69
        LIST = 70
        NOMC = 71
        NOMCSUBJ = 72
        NOMCSUBJPASS = 73
        NUMC = 74
        COP = 75
        DISLOCATED = 76
        ASP = 77
        GMOD = 78
        GOBJ = 79
        INFMOD = 80
        MES = 81
        NCOMP = 82
    headTokenIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    label = _messages.EnumField('LabelValueValuesEnum', 2)