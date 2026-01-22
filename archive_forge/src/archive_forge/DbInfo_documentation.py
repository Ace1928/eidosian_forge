import sys
from rdkit import RDConfig
from rdkit.Dbase import DbModule
 gets a list of columns available in a DB table

      **Arguments**

        - dBase: the name of the DB file to be used

        - table: the name of the table to query

        - user: the username for DB access

        - password: the password to be used for DB access

        - join: an optional join clause  (omit the verb 'join')

        - what: an optional clause indicating what to select

      **Returns**

        -  a list of column names

    