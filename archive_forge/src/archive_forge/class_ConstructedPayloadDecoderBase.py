import io
import os
import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.codec.streaming import asSeekableStream
from pyasn1.codec.streaming import isEndOfStream
from pyasn1.codec.streaming import peekIntoStream
from pyasn1.codec.streaming import readFromStream
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.error import PyAsn1Error
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful
class ConstructedPayloadDecoderBase(AbstractConstructedPayloadDecoder):
    protoRecordComponent = None
    protoSequenceComponent = None

    def _getComponentTagMap(self, asn1Object, idx):
        raise NotImplementedError()

    def _getComponentPositionByType(self, asn1Object, tagSet, idx):
        raise NotImplementedError()

    def _decodeComponentsSchemaless(self, substrate, tagSet=None, decodeFun=None, length=None, **options):
        asn1Object = None
        components = []
        componentTypes = set()
        original_position = substrate.tell()
        while length == -1 or substrate.tell() < original_position + length:
            for component in decodeFun(substrate, **options):
                if isinstance(component, SubstrateUnderrunError):
                    yield component
            if length == -1 and component is eoo.endOfOctets:
                break
            components.append(component)
            componentTypes.add(component.tagSet)
            if len(componentTypes) > 1:
                protoComponent = self.protoRecordComponent
            else:
                protoComponent = self.protoSequenceComponent
            asn1Object = protoComponent.clone(tagSet=tag.TagSet(protoComponent.tagSet.baseTag, *tagSet.superTags))
        if LOG:
            LOG('guessed %r container type (pass `asn1Spec` to guide the decoder)' % asn1Object)
        for idx, component in enumerate(components):
            asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
        yield asn1Object

    def valueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatConstructed:
            raise error.PyAsn1Error('Constructed tag format expected')
        original_position = substrate.tell()
        if substrateFun:
            if asn1Spec is not None:
                asn1Object = asn1Spec.clone()
            elif self.protoComponent is not None:
                asn1Object = self.protoComponent.clone(tagSet=tagSet)
            else:
                asn1Object = (self.protoRecordComponent, self.protoSequenceComponent)
            for chunk in substrateFun(asn1Object, substrate, length, options):
                yield chunk
            return
        if asn1Spec is None:
            for asn1Object in self._decodeComponentsSchemaless(substrate, tagSet=tagSet, decodeFun=decodeFun, length=length, **options):
                if isinstance(asn1Object, SubstrateUnderrunError):
                    yield asn1Object
            if substrate.tell() < original_position + length:
                if LOG:
                    for trailing in readFromStream(substrate, context=options):
                        if isinstance(trailing, SubstrateUnderrunError):
                            yield trailing
                    LOG('Unused trailing %d octets encountered: %s' % (len(trailing), debug.hexdump(trailing)))
            yield asn1Object
            return
        asn1Object = asn1Spec.clone()
        asn1Object.clear()
        options = self._passAsn1Object(asn1Object, options)
        if asn1Spec.typeId in (univ.Sequence.typeId, univ.Set.typeId):
            namedTypes = asn1Spec.componentType
            isSetType = asn1Spec.typeId == univ.Set.typeId
            isDeterministic = not isSetType and (not namedTypes.hasOptionalOrDefault)
            if LOG:
                LOG('decoding %sdeterministic %s type %r chosen by type ID' % (not isDeterministic and 'non-' or '', isSetType and 'SET' or '', asn1Spec))
            seenIndices = set()
            idx = 0
            while substrate.tell() - original_position < length:
                if not namedTypes:
                    componentType = None
                elif isSetType:
                    componentType = namedTypes.tagMapUnique
                else:
                    try:
                        if isDeterministic:
                            componentType = namedTypes[idx].asn1Object
                        elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                            componentType = namedTypes.getTagMapNearPosition(idx)
                        else:
                            componentType = namedTypes[idx].asn1Object
                    except IndexError:
                        raise error.PyAsn1Error('Excessive components decoded at %r' % (asn1Spec,))
                for component in decodeFun(substrate, componentType, **options):
                    if isinstance(component, SubstrateUnderrunError):
                        yield component
                if not isDeterministic and namedTypes:
                    if isSetType:
                        idx = namedTypes.getPositionByType(component.effectiveTagSet)
                    elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                        idx = namedTypes.getPositionNearType(component.effectiveTagSet, idx)
                asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
                seenIndices.add(idx)
                idx += 1
            if LOG:
                LOG('seen component indices %s' % seenIndices)
            if namedTypes:
                if not namedTypes.requiredComponents.issubset(seenIndices):
                    raise error.PyAsn1Error('ASN.1 object %s has uninitialized components' % asn1Object.__class__.__name__)
                if namedTypes.hasOpenTypes:
                    openTypes = options.get('openTypes', {})
                    if LOG:
                        LOG('user-specified open types map:')
                        for k, v in openTypes.items():
                            LOG('%s -> %r' % (k, v))
                    if openTypes or options.get('decodeOpenTypes', False):
                        for idx, namedType in enumerate(namedTypes.namedTypes):
                            if not namedType.openType:
                                continue
                            if namedType.isOptional and (not asn1Object.getComponentByPosition(idx).isValue):
                                continue
                            governingValue = asn1Object.getComponentByName(namedType.openType.name)
                            try:
                                openType = openTypes[governingValue]
                            except KeyError:
                                if LOG:
                                    LOG('default open types map of component "%s.%s" governed by component "%s.%s":' % (asn1Object.__class__.__name__, namedType.name, asn1Object.__class__.__name__, namedType.openType.name))
                                    for k, v in namedType.openType.items():
                                        LOG('%s -> %r' % (k, v))
                                try:
                                    openType = namedType.openType[governingValue]
                                except KeyError:
                                    if LOG:
                                        LOG('failed to resolve open type by governing value %r' % (governingValue,))
                                    continue
                            if LOG:
                                LOG('resolved open type %r by governing value %r' % (openType, governingValue))
                            containerValue = asn1Object.getComponentByPosition(idx)
                            if containerValue.typeId in (univ.SetOf.typeId, univ.SequenceOf.typeId):
                                for pos, containerElement in enumerate(containerValue):
                                    stream = asSeekableStream(containerValue[pos].asOctets())
                                    for component in decodeFun(stream, asn1Spec=openType, **options):
                                        if isinstance(component, SubstrateUnderrunError):
                                            yield component
                                    containerValue[pos] = component
                            else:
                                stream = asSeekableStream(asn1Object.getComponentByPosition(idx).asOctets())
                                for component in decodeFun(stream, asn1Spec=openType, **options):
                                    if isinstance(component, SubstrateUnderrunError):
                                        yield component
                                asn1Object.setComponentByPosition(idx, component)
            else:
                inconsistency = asn1Object.isInconsistent
                if inconsistency:
                    raise inconsistency
        else:
            componentType = asn1Spec.componentType
            if LOG:
                LOG('decoding type %r chosen by given `asn1Spec`' % componentType)
            idx = 0
            while substrate.tell() - original_position < length:
                for component in decodeFun(substrate, componentType, **options):
                    if isinstance(component, SubstrateUnderrunError):
                        yield component
                asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
                idx += 1
        yield asn1Object

    def indefLenValueDecoder(self, substrate, asn1Spec, tagSet=None, length=None, state=None, decodeFun=None, substrateFun=None, **options):
        if tagSet[0].tagFormat != tag.tagFormatConstructed:
            raise error.PyAsn1Error('Constructed tag format expected')
        if substrateFun is not None:
            if asn1Spec is not None:
                asn1Object = asn1Spec.clone()
            elif self.protoComponent is not None:
                asn1Object = self.protoComponent.clone(tagSet=tagSet)
            else:
                asn1Object = (self.protoRecordComponent, self.protoSequenceComponent)
            for chunk in substrateFun(asn1Object, substrate, length, options):
                yield chunk
            return
        if asn1Spec is None:
            for asn1Object in self._decodeComponentsSchemaless(substrate, tagSet=tagSet, decodeFun=decodeFun, length=length, **dict(options, allowEoo=True)):
                if isinstance(asn1Object, SubstrateUnderrunError):
                    yield asn1Object
            yield asn1Object
            return
        asn1Object = asn1Spec.clone()
        asn1Object.clear()
        options = self._passAsn1Object(asn1Object, options)
        if asn1Spec.typeId in (univ.Sequence.typeId, univ.Set.typeId):
            namedTypes = asn1Object.componentType
            isSetType = asn1Object.typeId == univ.Set.typeId
            isDeterministic = not isSetType and (not namedTypes.hasOptionalOrDefault)
            if LOG:
                LOG('decoding %sdeterministic %s type %r chosen by type ID' % (not isDeterministic and 'non-' or '', isSetType and 'SET' or '', asn1Spec))
            seenIndices = set()
            idx = 0
            while True:
                if len(namedTypes) <= idx:
                    asn1Spec = None
                elif isSetType:
                    asn1Spec = namedTypes.tagMapUnique
                else:
                    try:
                        if isDeterministic:
                            asn1Spec = namedTypes[idx].asn1Object
                        elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                            asn1Spec = namedTypes.getTagMapNearPosition(idx)
                        else:
                            asn1Spec = namedTypes[idx].asn1Object
                    except IndexError:
                        raise error.PyAsn1Error('Excessive components decoded at %r' % (asn1Object,))
                for component in decodeFun(substrate, asn1Spec, allowEoo=True, **options):
                    if isinstance(component, SubstrateUnderrunError):
                        yield component
                    if component is eoo.endOfOctets:
                        break
                if component is eoo.endOfOctets:
                    break
                if not isDeterministic and namedTypes:
                    if isSetType:
                        idx = namedTypes.getPositionByType(component.effectiveTagSet)
                    elif namedTypes[idx].isOptional or namedTypes[idx].isDefaulted:
                        idx = namedTypes.getPositionNearType(component.effectiveTagSet, idx)
                asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
                seenIndices.add(idx)
                idx += 1
            if LOG:
                LOG('seen component indices %s' % seenIndices)
            if namedTypes:
                if not namedTypes.requiredComponents.issubset(seenIndices):
                    raise error.PyAsn1Error('ASN.1 object %s has uninitialized components' % asn1Object.__class__.__name__)
                if namedTypes.hasOpenTypes:
                    openTypes = options.get('openTypes', {})
                    if LOG:
                        LOG('user-specified open types map:')
                        for k, v in openTypes.items():
                            LOG('%s -> %r' % (k, v))
                    if openTypes or options.get('decodeOpenTypes', False):
                        for idx, namedType in enumerate(namedTypes.namedTypes):
                            if not namedType.openType:
                                continue
                            if namedType.isOptional and (not asn1Object.getComponentByPosition(idx).isValue):
                                continue
                            governingValue = asn1Object.getComponentByName(namedType.openType.name)
                            try:
                                openType = openTypes[governingValue]
                            except KeyError:
                                if LOG:
                                    LOG('default open types map of component "%s.%s" governed by component "%s.%s":' % (asn1Object.__class__.__name__, namedType.name, asn1Object.__class__.__name__, namedType.openType.name))
                                    for k, v in namedType.openType.items():
                                        LOG('%s -> %r' % (k, v))
                                try:
                                    openType = namedType.openType[governingValue]
                                except KeyError:
                                    if LOG:
                                        LOG('failed to resolve open type by governing value %r' % (governingValue,))
                                    continue
                            if LOG:
                                LOG('resolved open type %r by governing value %r' % (openType, governingValue))
                            containerValue = asn1Object.getComponentByPosition(idx)
                            if containerValue.typeId in (univ.SetOf.typeId, univ.SequenceOf.typeId):
                                for pos, containerElement in enumerate(containerValue):
                                    stream = asSeekableStream(containerValue[pos].asOctets())
                                    for component in decodeFun(stream, asn1Spec=openType, **dict(options, allowEoo=True)):
                                        if isinstance(component, SubstrateUnderrunError):
                                            yield component
                                        if component is eoo.endOfOctets:
                                            break
                                    containerValue[pos] = component
                            else:
                                stream = asSeekableStream(asn1Object.getComponentByPosition(idx).asOctets())
                                for component in decodeFun(stream, asn1Spec=openType, **dict(options, allowEoo=True)):
                                    if isinstance(component, SubstrateUnderrunError):
                                        yield component
                                    if component is eoo.endOfOctets:
                                        break
                                    asn1Object.setComponentByPosition(idx, component)
                else:
                    inconsistency = asn1Object.isInconsistent
                    if inconsistency:
                        raise inconsistency
        else:
            componentType = asn1Spec.componentType
            if LOG:
                LOG('decoding type %r chosen by given `asn1Spec`' % componentType)
            idx = 0
            while True:
                for component in decodeFun(substrate, componentType, allowEoo=True, **options):
                    if isinstance(component, SubstrateUnderrunError):
                        yield component
                    if component is eoo.endOfOctets:
                        break
                if component is eoo.endOfOctets:
                    break
                asn1Object.setComponentByPosition(idx, component, verifyConstraints=False, matchTags=False, matchConstraints=False)
                idx += 1
        yield asn1Object