import os
from sqlalchemy import (Column, Float, Integer, LargeBinary, String, Text,
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
import rdkit.RDLogger as logging
def RegisterSchema(dbUrl, echo=False):
    engine = create_engine(dbUrl, echo=echo)
    decBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)