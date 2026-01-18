import copy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
def transferAgentsToReactants(rxn):
    for a in range(rxn.GetNumAgentTemplates()):
        agent = rxn.GetAgentTemplate(a)
        rxn.AddReactantTemplate(agent)