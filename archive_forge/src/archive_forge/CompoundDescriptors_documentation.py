from rdkit import RDConfig
from rdkit.ML.Descriptors import Descriptors, Parser
from rdkit.utils import chemutils
 Constructor

      **Arguments**

        - simpleList: list of simple descriptors to be calculated
              (see below for format)

        - compoundList: list of compound descriptors to be calculated
              (see below for format)

        - dbName: name of the atomic database to be used

        - dbTable: name the table in _dbName_ which has atomic data

        - dbUser: user name for DB access

        - dbPassword: password for DB access

      **Note**

        - format of simpleList:
           a list of 2-tuples containing:

              1) name of the atomic descriptor

              2) a list of operations on that descriptor (e.g. NonZero, Max, etc.)
                 These must correspond to the *Calculator Method* names above.

        - format of compoundList:
           a list of 2-tuples containing:

              1) name of the descriptor to be calculated

              2) list of selected atomic descriptor names (define $1, $2, etc.)

              3) list of selected compound descriptor names (define $a, $b, etc.)

              4) text formula defining the calculation (see _Parser_)

    