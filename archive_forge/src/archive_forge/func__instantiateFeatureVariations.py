from fontTools.ttLib.tables import otTables as ot
from copy import deepcopy
import logging
def _instantiateFeatureVariations(table, fvarAxes, axisLimits):
    pinnedAxes = set(axisLimits.pinnedLocation())
    axisOrder = [axis.axisTag for axis in fvarAxes if axis.axisTag not in pinnedAxes]
    axisIndexMap = {axisTag: axisOrder.index(axisTag) for axisTag in axisOrder}
    featureVariationApplied = False
    uniqueRecords = set()
    newRecords = []
    defaultsSubsts = None
    for i, record in enumerate(table.FeatureVariations.FeatureVariationRecord):
        applies, shouldKeep, universal = _instantiateFeatureVariationRecord(record, i, axisLimits, fvarAxes, axisIndexMap)
        if shouldKeep and _featureVariationRecordIsUnique(record, uniqueRecords):
            newRecords.append(record)
        if applies and (not featureVariationApplied):
            assert record.FeatureTableSubstitution.Version == 65536
            defaultsSubsts = deepcopy(record.FeatureTableSubstitution)
            for default, rec in zip(defaultsSubsts.SubstitutionRecord, record.FeatureTableSubstitution.SubstitutionRecord):
                default.Feature = deepcopy(table.FeatureList.FeatureRecord[rec.FeatureIndex].Feature)
                table.FeatureList.FeatureRecord[rec.FeatureIndex].Feature = deepcopy(rec.Feature)
            featureVariationApplied = True
        if universal:
            break
    if featureVariationApplied and newRecords and (not universal):
        defaultRecord = ot.FeatureVariationRecord()
        defaultRecord.ConditionSet = ot.ConditionSet()
        defaultRecord.ConditionSet.ConditionTable = []
        defaultRecord.ConditionSet.ConditionCount = 0
        defaultRecord.FeatureTableSubstitution = defaultsSubsts
        newRecords.append(defaultRecord)
    if newRecords:
        table.FeatureVariations.FeatureVariationRecord = newRecords
        table.FeatureVariations.FeatureVariationCount = len(newRecords)
    else:
        del table.FeatureVariations
        table.Version = 65536