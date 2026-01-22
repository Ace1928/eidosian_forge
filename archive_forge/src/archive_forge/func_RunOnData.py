from warnings import warn
import pickle
import sys
import time
import numpy
from rdkit import DataStructs
from rdkit.Dbase import DbModule
from rdkit.ML import CompositeRun, ScreenComposite
from rdkit.ML.Composite import BayesComposite, Composite
from rdkit.ML.Data import DataUtils, SplitData
from rdkit.utils import listutils
def RunOnData(details, data, progressCallback=None, saveIt=1, setDescNames=0):
    if details.lockRandom:
        seed = details.randomSeed
    else:
        import random
        seed = (random.randint(0, 1000000.0), random.randint(0, 1000000.0))
    DataUtils.InitRandomNumbers(seed)
    testExamples = []
    if details.shuffleActivities == 1:
        DataUtils.RandomizeActivities(data, shuffle=1, runDetails=details)
    elif details.randomActivities == 1:
        DataUtils.RandomizeActivities(data, shuffle=0, runDetails=details)
    namedExamples = data.GetNamedData()
    if details.splitRun == 1:
        trainIdx, testIdx = SplitData.SplitIndices(len(namedExamples), details.splitFrac, silent=not _verbose)
        trainExamples = [namedExamples[x] for x in trainIdx]
        testExamples = [namedExamples[x] for x in testIdx]
    else:
        testExamples = []
        testIdx = []
        trainIdx = list(range(len(namedExamples)))
        trainExamples = namedExamples
    if details.filterFrac != 0.0:
        if hasattr(details, 'activityBounds') and details.activityBounds:
            tExamples = []
            bounds = details.activityBounds
            for pt in trainExamples:
                pt = pt[:]
                act = pt[-1]
                placed = 0
                bound = 0
                while not placed and bound < len(bounds):
                    if act < bounds[bound]:
                        pt[-1] = bound
                        placed = 1
                    else:
                        bound += 1
                if not placed:
                    pt[-1] = bound
                tExamples.append(pt)
        else:
            bounds = None
            tExamples = trainExamples
        trainIdx, temp = DataUtils.FilterData(tExamples, details.filterVal, details.filterFrac, -1, indicesOnly=1)
        tmp = [trainExamples[x] for x in trainIdx]
        testExamples += [trainExamples[x] for x in temp]
        trainExamples = tmp
        counts = DataUtils.CountResults(trainExamples, bounds=bounds)
        ks = counts.keys()
        ks.sort()
        message('Result Counts in training set:')
        for k in ks:
            message(str((k, counts[k])))
        counts = DataUtils.CountResults(testExamples, bounds=bounds)
        ks = counts.keys()
        ks.sort()
        message('Result Counts in test set:')
        for k in ks:
            message(str((k, counts[k])))
    nExamples = len(trainExamples)
    message('Training with %d examples' % nExamples)
    nVars = data.GetNVars()
    attrs = list(range(1, nVars + 1))
    nPossibleVals = data.GetNPossibleVals()
    for i in range(1, len(nPossibleVals)):
        if nPossibleVals[i - 1] == -1:
            attrs.remove(i)
    if details.pickleDataFileName != '':
        pickleDataFile = open(details.pickleDataFileName, 'wb+')
        pickle.dump(trainExamples, pickleDataFile)
        pickle.dump(testExamples, pickleDataFile)
        pickleDataFile.close()
    if details.bayesModel:
        composite = BayesComposite.BayesComposite()
    else:
        composite = Composite.Composite()
    composite._randomSeed = seed
    composite._splitFrac = details.splitFrac
    composite._shuffleActivities = details.shuffleActivities
    composite._randomizeActivities = details.randomActivities
    if hasattr(details, 'filterFrac'):
        composite._filterFrac = details.filterFrac
    if hasattr(details, 'filterVal'):
        composite._filterVal = details.filterVal
    composite.SetModelFilterData(details.modelFilterFrac, details.modelFilterVal)
    composite.SetActivityQuantBounds(details.activityBounds)
    nPossibleVals = data.GetNPossibleVals()
    if details.activityBounds:
        nPossibleVals[-1] = len(details.activityBounds) + 1
    if setDescNames:
        composite.SetInputOrder(data.GetVarNames())
        composite.SetDescriptorNames(details._descNames)
    else:
        composite.SetDescriptorNames(data.GetVarNames())
    composite.SetActivityQuantBounds(details.activityBounds)
    if details.nModels == 1:
        details.internalHoldoutFrac = 0.0
    if details.useTrees:
        from rdkit.ML.DecTree import CrossValidate, PruneTree
        if details.qBounds != []:
            from rdkit.ML.DecTree import BuildQuantTree
            builder = BuildQuantTree.QuantTreeBoot
        else:
            from rdkit.ML.DecTree import ID3
            builder = ID3.ID3Boot
        driver = CrossValidate.CrossValidationDriver
        pruner = PruneTree.PruneTree
        composite.SetQuantBounds(details.qBounds)
        nPossibleVals = data.GetNPossibleVals()
        if details.activityBounds:
            nPossibleVals[-1] = len(details.activityBounds) + 1
        composite.Grow(trainExamples, attrs, nPossibleVals=[0] + nPossibleVals, buildDriver=driver, pruner=pruner, nTries=details.nModels, pruneIt=details.pruneIt, lessGreedy=details.lessGreedy, needsQuantization=0, treeBuilder=builder, nQuantBounds=details.qBounds, startAt=details.startAt, maxDepth=details.limitDepth, progressCallback=progressCallback, holdOutFrac=details.internalHoldoutFrac, replacementSelection=details.replacementSelection, recycleVars=details.recycleVars, randomDescriptors=details.randomDescriptors, silent=not _verbose)
    elif details.useSigTrees:
        from rdkit.ML.DecTree import BuildSigTree, CrossValidate
        builder = BuildSigTree.SigTreeBuilder
        driver = CrossValidate.CrossValidationDriver
        nPossibleVals = data.GetNPossibleVals()
        if details.activityBounds:
            nPossibleVals[-1] = len(details.activityBounds) + 1
        if hasattr(details, 'sigTreeBiasList'):
            biasList = details.sigTreeBiasList
        else:
            biasList = None
        if hasattr(details, 'useCMIM'):
            useCMIM = details.useCMIM
        else:
            useCMIM = 0
        if hasattr(details, 'allowCollections'):
            allowCollections = details.allowCollections
        else:
            allowCollections = False
        composite.Grow(trainExamples, attrs, nPossibleVals=[0] + nPossibleVals, buildDriver=driver, nTries=details.nModels, needsQuantization=0, treeBuilder=builder, maxDepth=details.limitDepth, progressCallback=progressCallback, holdOutFrac=details.internalHoldoutFrac, replacementSelection=details.replacementSelection, recycleVars=details.recycleVars, randomDescriptors=details.randomDescriptors, biasList=biasList, useCMIM=useCMIM, allowCollection=allowCollections, silent=not _verbose)
    elif details.useKNN:
        from rdkit.ML.KNN import CrossValidate, DistFunctions
        driver = CrossValidate.CrossValidationDriver
        dfunc = ''
        if details.knnDistFunc == 'Euclidean':
            dfunc = DistFunctions.EuclideanDist
        elif details.knnDistFunc == 'Tanimoto':
            dfunc = DistFunctions.TanimotoDist
        else:
            assert 0, 'Bad KNN distance metric value'
        composite.Grow(trainExamples, attrs, nPossibleVals=[0] + nPossibleVals, buildDriver=driver, nTries=details.nModels, needsQuantization=0, numNeigh=details.knnNeighs, holdOutFrac=details.internalHoldoutFrac, distFunc=dfunc)
    elif details.useNaiveBayes or details.useSigBayes:
        from rdkit.ML.NaiveBayes import CrossValidate
        driver = CrossValidate.CrossValidationDriver
        if not (hasattr(details, 'useSigBayes') and details.useSigBayes):
            composite.Grow(trainExamples, attrs, nPossibleVals=[0] + nPossibleVals, buildDriver=driver, nTries=details.nModels, needsQuantization=0, nQuantBounds=details.qBounds, holdOutFrac=details.internalHoldoutFrac, replacementSelection=details.replacementSelection, mEstimateVal=details.mEstimateVal, silent=not _verbose)
        else:
            if hasattr(details, 'useCMIM'):
                useCMIM = details.useCMIM
            else:
                useCMIM = 0
            composite.Grow(trainExamples, attrs, nPossibleVals=[0] + nPossibleVals, buildDriver=driver, nTries=details.nModels, needsQuantization=0, nQuantBounds=details.qBounds, mEstimateVal=details.mEstimateVal, useSigs=True, useCMIM=useCMIM, holdOutFrac=details.internalHoldoutFrac, replacementSelection=details.replacementSelection, silent=not _verbose)
    else:
        from rdkit.ML.Neural import CrossValidate
        driver = CrossValidate.CrossValidationDriver
        composite.Grow(trainExamples, attrs, [0] + nPossibleVals, nTries=details.nModels, buildDriver=driver, needsQuantization=0)
    composite.AverageErrors()
    composite.SortModels()
    modelList, counts, avgErrs = composite.GetAllData()
    counts = numpy.array(counts)
    avgErrs = numpy.array(avgErrs)
    composite._varNames = data.GetVarNames()
    for i in range(len(modelList)):
        modelList[i].NameModel(composite._varNames)
    weightedErrs = counts * avgErrs
    averageErr = sum(weightedErrs) / sum(counts)
    devs = avgErrs - averageErr
    devs = devs * counts
    devs = numpy.sqrt(devs * devs)
    avgDev = sum(devs) / sum(counts)
    message('# Overall Average Error: %%% 5.2f, Average Deviation: %%% 6.2f' % (100.0 * averageErr, 100.0 * avgDev))
    if details.bayesModel:
        composite.Train(trainExamples, verbose=0)
    composite.ClearModelExamples()
    if saveIt:
        composite.Pickle(details.outName)
    details.model = DbModule.binaryHolder(pickle.dumps(composite))
    badExamples = []
    if not details.detailedRes and (not hasattr(details, 'noScreen') or not details.noScreen):
        if details.splitRun:
            message('Testing all hold-out examples')
            wrong = testall(composite, testExamples, badExamples)
            message('%d examples (%% %5.2f) were misclassified' % (len(wrong), 100.0 * float(len(wrong)) / float(len(testExamples))))
            _runDetails.holdout_error = float(len(wrong)) / len(testExamples)
        else:
            message('Testing all examples')
            wrong = testall(composite, namedExamples, badExamples)
            message('%d examples (%% %5.2f) were misclassified' % (len(wrong), 100.0 * float(len(wrong)) / float(len(namedExamples))))
            _runDetails.overall_error = float(len(wrong)) / len(namedExamples)
    if details.detailedRes:
        message('\nEntire data set:')
        resTup = ScreenComposite.ShowVoteResults(range(data.GetNPts()), data, composite, nPossibleVals[-1], details.threshold)
        nGood, nBad, nSkip, avgGood, avgBad, avgSkip, voteTab = resTup
        nPts = len(namedExamples)
        nClass = nGood + nBad
        _runDetails.overall_error = float(nBad) / nClass
        _runDetails.overall_correct_conf = avgGood
        _runDetails.overall_incorrect_conf = avgBad
        _runDetails.overall_result_matrix = repr(voteTab)
        nRej = nClass - nPts
        if nRej > 0:
            _runDetails.overall_fraction_dropped = float(nRej) / nPts
        if details.splitRun:
            message('\nHold-out data:')
            resTup = ScreenComposite.ShowVoteResults(range(len(testExamples)), testExamples, composite, nPossibleVals[-1], details.threshold)
            nGood, nBad, nSkip, avgGood, avgBad, avgSkip, voteTab = resTup
            nPts = len(testExamples)
            nClass = nGood + nBad
            _runDetails.holdout_error = float(nBad) / nClass
            _runDetails.holdout_correct_conf = avgGood
            _runDetails.holdout_incorrect_conf = avgBad
            _runDetails.holdout_result_matrix = repr(voteTab)
            nRej = nClass - nPts
            if nRej > 0:
                _runDetails.holdout_fraction_dropped = float(nRej) / nPts
    if details.persistTblName and details.dbName:
        message('Updating results table %s:%s' % (details.dbName, details.persistTblName))
        details.Store(db=details.dbName, table=details.persistTblName)
    if details.badName != '':
        badFile = open(details.badName, 'w+')
        for i in range(len(badExamples)):
            ex = badExamples[i]
            vote = wrong[i]
            outStr = '%s\t%s\n' % (ex, vote)
            badFile.write(outStr)
        badFile.close()
    composite.ClearModelExamples()
    return composite