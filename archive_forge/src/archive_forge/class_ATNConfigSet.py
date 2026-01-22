from io import StringIO
from functools import reduce
from antlr4.PredictionContext import PredictionContext, merge
from antlr4.Utils import str_list
from antlr4.atn.ATN import ATN
from antlr4.atn.ATNConfig import ATNConfig
from antlr4.atn.SemanticContext import SemanticContext
from antlr4.error.Errors import UnsupportedOperationException, IllegalStateException
class ATNConfigSet(object):
    __slots__ = ('configLookup', 'fullCtx', 'readonly', 'configs', 'uniqueAlt', 'conflictingAlts', 'hasSemanticContext', 'dipsIntoOuterContext', 'cachedHashCode')

    def __init__(self, fullCtx: bool=True):
        self.configLookup = dict()
        self.fullCtx = fullCtx
        self.readonly = False
        self.configs = []
        self.uniqueAlt = 0
        self.conflictingAlts = None
        self.hasSemanticContext = False
        self.dipsIntoOuterContext = False
        self.cachedHashCode = -1

    def __iter__(self):
        return self.configs.__iter__()

    def add(self, config: ATNConfig, mergeCache=None):
        if self.readonly:
            raise Exception('This set is readonly')
        if config.semanticContext is not SemanticContext.NONE:
            self.hasSemanticContext = True
        if config.reachesIntoOuterContext > 0:
            self.dipsIntoOuterContext = True
        existing = self.getOrAdd(config)
        if existing is config:
            self.cachedHashCode = -1
            self.configs.append(config)
            return True
        rootIsWildcard = not self.fullCtx
        merged = merge(existing.context, config.context, rootIsWildcard, mergeCache)
        existing.reachesIntoOuterContext = max(existing.reachesIntoOuterContext, config.reachesIntoOuterContext)
        if config.precedenceFilterSuppressed:
            existing.precedenceFilterSuppressed = True
        existing.context = merged
        return True

    def getOrAdd(self, config: ATNConfig):
        h = config.hashCodeForConfigSet()
        l = self.configLookup.get(h, None)
        if l is not None:
            r = next((cfg for cfg in l if config.equalsForConfigSet(cfg)), None)
            if r is not None:
                return r
        if l is None:
            l = [config]
            self.configLookup[h] = l
        else:
            l.append(config)
        return config

    def getStates(self):
        return set((c.state for c in self.configs))

    def getPredicates(self):
        return list((cfg.semanticContext for cfg in self.configs if cfg.semanticContext != SemanticContext.NONE))

    def get(self, i: int):
        return self.configs[i]

    def optimizeConfigs(self, interpreter: ATNSimulator):
        if self.readonly:
            raise IllegalStateException('This set is readonly')
        if len(self.configs) == 0:
            return
        for config in self.configs:
            config.context = interpreter.getCachedContext(config.context)

    def addAll(self, coll: list):
        for c in coll:
            self.add(c)
        return False

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, ATNConfigSet):
            return False
        same = self.configs is not None and self.configs == other.configs and (self.fullCtx == other.fullCtx) and (self.uniqueAlt == other.uniqueAlt) and (self.conflictingAlts == other.conflictingAlts) and (self.hasSemanticContext == other.hasSemanticContext) and (self.dipsIntoOuterContext == other.dipsIntoOuterContext)
        return same

    def __hash__(self):
        if self.readonly:
            if self.cachedHashCode == -1:
                self.cachedHashCode = self.hashConfigs()
            return self.cachedHashCode
        return self.hashConfigs()

    def hashConfigs(self):
        return reduce(lambda h, cfg: hash((h, cfg)), self.configs, 0)

    def __len__(self):
        return len(self.configs)

    def isEmpty(self):
        return len(self.configs) == 0

    def __contains__(self, config):
        if self.configLookup is None:
            raise UnsupportedOperationException('This method is not implemented for readonly sets.')
        h = config.hashCodeForConfigSet()
        l = self.configLookup.get(h, None)
        if l is not None:
            for c in l:
                if config.equalsForConfigSet(c):
                    return True
        return False

    def clear(self):
        if self.readonly:
            raise IllegalStateException('This set is readonly')
        self.configs.clear()
        self.cachedHashCode = -1
        self.configLookup.clear()

    def setReadonly(self, readonly: bool):
        self.readonly = readonly
        self.configLookup = None

    def __str__(self):
        with StringIO() as buf:
            buf.write(str_list(self.configs))
            if self.hasSemanticContext:
                buf.write(',hasSemanticContext=')
                buf.write(str(self.hasSemanticContext))
            if self.uniqueAlt != ATN.INVALID_ALT_NUMBER:
                buf.write(',uniqueAlt=')
                buf.write(str(self.uniqueAlt))
            if self.conflictingAlts is not None:
                buf.write(',conflictingAlts=')
                buf.write(str(self.conflictingAlts))
            if self.dipsIntoOuterContext:
                buf.write(',dipsIntoOuterContext')
            return buf.getvalue()