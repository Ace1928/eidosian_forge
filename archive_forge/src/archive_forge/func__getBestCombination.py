import itertools
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
from . import utils
def _getBestCombination(rfps, pfps, output=False):
    if output:
        print('--- _getBestCombination ---')
    tests = []
    numReactants = len(rfps)
    for i in range(1, numReactants + 1):
        for x in itertools.combinations(range(numReactants), i):
            temp = []
            for j in x:
                if not rfps[j][1]:
                    numAtms = rfps[j][0].molecule.GetNumAtoms()
                    if numAtms > 1:
                        temp.append((rfps[j][0].molecule.GetNumAtoms(), j))
                elif output > 3:
                    print('Frequent reagent found: ', j)
            if temp not in tests:
                tests.append(temp)
    maxScore = 0
    maxDetailScore = 0
    finalReacts = [[]]
    productsDetailFP = utils.getSumFps([i.detailFP for i in pfps])
    productsScaffoldFP = utils.getSumFps([i.scaffoldFP for i in pfps])
    numProductAtoms = 0
    for i in pfps:
        numProductAtoms += i.molecule.GetNumAtoms()
    productsDetailFPBitInfo = {}
    productsScaffoldFPBitInfo = {}
    for i in pfps:
        productsDetailFPBitInfo.update(i.bitInfoDetailFP)
        productsScaffoldFPBitInfo.update(i.bitInfoScaffoldFP)
    numUnmappedPAtoms, bitsUnmappedPAtoms = utils.getNumPositiveBitCountsOfRadius0(productsScaffoldFP, productsScaffoldFPBitInfo)
    finalNumUnmappedProdAtoms = [[len(productsDetailFP.GetNonzeroElements()), len(productsScaffoldFP.GetNonzeroElements()), numUnmappedPAtoms, bitsUnmappedPAtoms]]
    for test in tests:
        if len(test) < 1:
            continue
        numReactantAtoms = np.array(test)[:, 0].sum()
        if numReactantAtoms > 5 * numProductAtoms or numReactantAtoms < numProductAtoms * 0.8:
            continue
        if output > 0:
            print('Combination: ', test)
        reactantsDetailFP = utils.getSumFps([rfps[i[1]][0].detailFP for i in test])
        reactantsScaffoldFP = utils.getSumFps([rfps[i[1]][0].scaffoldFP for i in test])
        detailFPScore = _calcScore(reactantsDetailFP, productsDetailFP, bitInfoProd=productsDetailFPBitInfo, output=output)
        scaffoldFPScore = _calcScore(reactantsScaffoldFP, productsScaffoldFP, bitInfoProd=productsScaffoldFPBitInfo, output=output)
        score = detailFPScore[0] + scaffoldFPScore[0]
        if output > 0:
            print('>>>> score: ', score)
            print('>>>> scores (detail, scaffold): ', detailFPScore[0], scaffoldFPScore[0])
            print('>>>> num unmapped productFP bits: ', detailFPScore[1], scaffoldFPScore[1], detailFPScore[2], scaffoldFPScore[2])
        if score > maxScore:
            maxScore = score
            maxDetailScore = detailFPScore[0]
            del finalReacts[:]
            del finalNumUnmappedProdAtoms[:]
            finalReacts.append([i[1] for i in test])
            finalNumUnmappedProdAtoms.append([detailFPScore[1], scaffoldFPScore[2], scaffoldFPScore[1], scaffoldFPScore[-1]])
            if output > 0:
                print(' >> maxScore: ', maxScore)
                print(' >> Final reactants: ', finalReacts)
            if scaffoldFPScore[0] > 0.9999 and detailFPScore[0] > 0.8:
                return (finalReacts, finalNumUnmappedProdAtoms)
            if len(finalNumUnmappedProdAtoms) > 0 and len(test) == 1:
                if finalNumUnmappedProdAtoms[0][1] == 0 and finalNumUnmappedProdAtoms[0][0] <= 3:
                    return (finalReacts, finalNumUnmappedProdAtoms)
        elif abs(score - maxScore) < 1e-07 and score > 0.0:
            finalReacts.append([i[1] for i in test])
            finalNumUnmappedProdAtoms.append([detailFPScore[1], scaffoldFPScore[2], scaffoldFPScore[1], scaffoldFPScore[-1]])
            if output > 0:
                print(' >> Added alternative result')
                print(' >> Final reactants: ', finalReacts)
    return (finalReacts, finalNumUnmappedProdAtoms)