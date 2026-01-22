import datetime
import operator
from functools import reduce
from zope.interface import implementer
from constantly import ValueConstant, Values
from twisted.positioning import _sentence, base, ipositioning
from twisted.positioning.base import Angles
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import iterbytes, nativeString
@implementer(ipositioning.INMEAReceiver)
class NMEAAdapter:
    """
    An adapter from NMEAProtocol receivers to positioning receivers.

    @cvar _STATEFUL_UPDATE: Information on how to update partial information
        in the sentence data or internal adapter state. For more information,
        see C{_statefulUpdate}'s docstring.
    @type _STATEFUL_UPDATE: See C{_statefulUpdate}'s docstring
    @cvar _ACCEPTABLE_UNITS: A set of NMEA notations of units that are
        already acceptable (metric), and therefore don't need to be converted.
    @type _ACCEPTABLE_UNITS: C{frozenset} of bytestrings
    @cvar _UNIT_CONVERTERS: Mapping of NMEA notations of units that are not
        acceptable (not metric) to converters that take a quantity in that
        unit and produce a metric quantity.
    @type _UNIT_CONVERTERS: C{dict} of bytestrings to unary callables
    @cvar  _SPECIFIC_SENTENCE_FIXES: A mapping of sentece types to specific
        fixes that are required to extract useful information from data from
        those sentences.
    @type  _SPECIFIC_SENTENCE_FIXES: C{dict} of sentence types to callables
        that take self and modify it in-place
    @cvar _FIXERS: Set of unary callables that take an NMEAAdapter instance
        and extract useful data from the sentence data, usually modifying the
        adapter's sentence data in-place.
    @type _FIXERS: C{dict} of native strings to unary callables
    @ivar yearThreshold: The earliest possible year that data will be
        interpreted as. For example, if this value is C{1990}, an NMEA
        0183 two-digit year of "96" will be interpreted as 1996, and
        a two-digit year of "13" will be interpreted as 2013.
    @type yearThreshold: L{int}
    @ivar _state: The current internal state of the receiver.
    @type _state: C{dict}
    @ivar _sentenceData: The data present in the sentence currently being
        processed. Starts empty, is filled as the sentence is parsed.
    @type _sentenceData: C{dict}
    @ivar _receiver: The positioning receiver that will receive parsed data.
    @type _receiver: L{ipositioning.IPositioningReceiver}
    """

    def __init__(self, receiver):
        """
        Initializes a new NMEA adapter.

        @param receiver: The receiver for positioning sentences.
        @type receiver: L{ipositioning.IPositioningReceiver}
        """
        self._state = {}
        self._sentenceData = {}
        self._receiver = receiver

    def _fixTimestamp(self):
        """
        Turns the NMEAProtocol timestamp notation into a datetime.time object.
        The time in this object is expressed as Zulu time.
        """
        timestamp = self.currentSentence.timestamp.split('.')[0]
        timeObject = datetime.datetime.strptime(timestamp, '%H%M%S').time()
        self._sentenceData['_time'] = timeObject
    yearThreshold = 1980

    def _fixDatestamp(self):
        """
        Turns an NMEA datestamp format into a C{datetime.date} object.

        @raise ValueError: When the day or month value was invalid, e.g. 32nd
            day, or 13th month, or 0th day or month.
        """
        date = self.currentSentence.datestamp
        day, month, year = map(int, [date[0:2], date[2:4], date[4:6]])
        year += self.yearThreshold - self.yearThreshold % 100
        if year < self.yearThreshold:
            year += 100
        self._sentenceData['_date'] = datetime.date(year, month, day)

    def _fixCoordinateFloat(self, coordinateType):
        """
        Turns the NMEAProtocol coordinate format into Python float.

        @param coordinateType: The coordinate type.
        @type coordinateType: One of L{Angles.LATITUDE} or L{Angles.LONGITUDE}.
        """
        if coordinateType is Angles.LATITUDE:
            coordinateName = 'latitude'
        else:
            coordinateName = 'longitude'
        nmeaCoordinate = getattr(self.currentSentence, coordinateName + 'Float')
        left, right = nmeaCoordinate.split('.')
        degrees, minutes = (int(left[:-2]), float(f'{left[-2:]}.{right}'))
        angle = degrees + minutes / 60
        coordinate = base.Coordinate(angle, coordinateType)
        self._sentenceData[coordinateName] = coordinate

    def _fixHemisphereSign(self, coordinateType, sentenceDataKey=None):
        """
        Fixes the sign for a hemisphere.

        This method must be called after the magnitude for the thing it
        determines the sign of has been set. This is done by the following
        functions:

            - C{self.FIXERS['magneticVariation']}
            - C{self.FIXERS['latitudeFloat']}
            - C{self.FIXERS['longitudeFloat']}

        @param coordinateType: Coordinate type. One of L{Angles.LATITUDE},
            L{Angles.LONGITUDE} or L{Angles.VARIATION}.
        @param sentenceDataKey: The key name of the hemisphere sign being
            fixed in the sentence data. If unspecified, C{coordinateType} is
            used.
        @type sentenceDataKey: C{str} (unless L{None})
        """
        sentenceDataKey = sentenceDataKey or coordinateType
        sign = self._getHemisphereSign(coordinateType)
        self._sentenceData[sentenceDataKey].setSign(sign)

    def _getHemisphereSign(self, coordinateType):
        """
        Returns the hemisphere sign for a given coordinate type.

        @param coordinateType: The coordinate type to find the hemisphere for.
        @type coordinateType: L{Angles.LATITUDE}, L{Angles.LONGITUDE} or
            L{Angles.VARIATION}.
        @return: The sign of that hemisphere (-1 or 1).
        @rtype: C{int}
        """
        if coordinateType is Angles.LATITUDE:
            hemisphereKey = 'latitudeHemisphere'
        elif coordinateType is Angles.LONGITUDE:
            hemisphereKey = 'longitudeHemisphere'
        elif coordinateType is Angles.VARIATION:
            hemisphereKey = 'magneticVariationDirection'
        else:
            raise ValueError(f'unknown coordinate type {coordinateType}')
        hemisphere = getattr(self.currentSentence, hemisphereKey).upper()
        if hemisphere in 'NE':
            return 1
        elif hemisphere in 'SW':
            return -1
        else:
            raise ValueError(f'bad hemisphere/direction: {hemisphere}')

    def _convert(self, key, converter):
        """
        A simple conversion fix.

        @param key: The attribute name of the value to fix.
        @type key: native string (Python identifier)

        @param converter: The function that converts the value.
        @type converter: unary callable
        """
        currentValue = getattr(self.currentSentence, key)
        self._sentenceData[key] = converter(currentValue)
    _STATEFUL_UPDATE = {'trueHeading': ('heading', base.Heading, '_angle', float), 'magneticVariation': ('heading', base.Heading, 'variation', lambda angle: base.Angle(float(angle), Angles.VARIATION)), 'horizontalDilutionOfPrecision': ('positionError', base.PositionError, 'hdop', float), 'verticalDilutionOfPrecision': ('positionError', base.PositionError, 'vdop', float), 'positionDilutionOfPrecision': ('positionError', base.PositionError, 'pdop', float)}

    def _statefulUpdate(self, sentenceKey):
        """
        Does a stateful update of a particular positioning attribute.
        Specifically, this will mutate an object in the current sentence data.

        Using the C{sentenceKey}, this will get a tuple containing, in order,
        the key name in the current state and sentence data, a factory for
        new values, the attribute to update, and a converter from sentence
        data (in NMEA notation) to something useful.

        If the sentence data doesn't have this data yet, it is grabbed from
        the state. If that doesn't have anything useful yet either, the
        factory is called to produce a new, empty object. Either way, the
        object ends up in the sentence data.

        @param sentenceKey: The name of the key in the sentence attributes,
            C{NMEAAdapter._STATEFUL_UPDATE} dictionary and the adapter state.
        @type sentenceKey: C{str}
        """
        key, factory, attr, converter = self._STATEFUL_UPDATE[sentenceKey]
        if key not in self._sentenceData:
            try:
                self._sentenceData[key] = self._state[key]
            except KeyError:
                self._sentenceData[key] = factory()
        newValue = converter(getattr(self.currentSentence, sentenceKey))
        setattr(self._sentenceData[key], attr, newValue)
    _ACCEPTABLE_UNITS = frozenset(['M'])
    _UNIT_CONVERTERS = {'N': lambda inKnots: base.Speed(float(inKnots) * base.MPS_PER_KNOT), 'K': lambda inKPH: base.Speed(float(inKPH) * base.MPS_PER_KPH)}

    def _fixUnits(self, unitKey=None, valueKey=None, sourceKey=None, unit=None):
        """
        Fixes the units of a certain value. If the units are already
        acceptable (metric), does nothing.

        None of the keys are allowed to be the empty string.

        @param unit: The unit that is being converted I{from}. If unspecified
            or L{None}, asks the current sentence for the C{unitKey}. If that
            also fails, raises C{AttributeError}.
        @type unit: C{str}
        @param unitKey: The name of the key/attribute under which the unit can
            be found in the current sentence. If the C{unit} parameter is set,
            this parameter is not used.
        @type unitKey: C{str}
        @param sourceKey: The name of the key/attribute that contains the
            current value to be converted (expressed in units as defined
            according to the C{unit} parameter). If unset, will use the
            same key as the value key.
        @type sourceKey: C{str}
        @param valueKey: The key name in which the data will be stored in the
            C{_sentenceData} instance attribute. If unset, attempts to remove
            "Units" from the end of the C{unitKey} parameter. If that fails,
            raises C{ValueError}.
        @type valueKey: C{str}
        """
        if unit is None:
            unit = getattr(self.currentSentence, unitKey)
        if valueKey is None:
            if unitKey is not None and unitKey.endswith('Units'):
                valueKey = unitKey[:-5]
            else:
                raise ValueError("valueKey unspecified and couldn't be guessed")
        if sourceKey is None:
            sourceKey = valueKey
        if unit not in self._ACCEPTABLE_UNITS:
            converter = self._UNIT_CONVERTERS[unit]
            currentValue = getattr(self.currentSentence, sourceKey)
            self._sentenceData[valueKey] = converter(currentValue)

    def _fixGSV(self):
        """
        Parses partial visible satellite information from a GSV sentence.
        """
        beaconInformation = base.BeaconInformation()
        self._sentenceData['_partialBeaconInformation'] = beaconInformation
        keys = ('satellitePRN', 'azimuth', 'elevation', 'signalToNoiseRatio')
        for index in range(4):
            prn, azimuth, elevation, snr = (getattr(self.currentSentence, attr) for attr in ('%s_%i' % (key, index) for key in keys))
            if prn is None or snr is None:
                continue
            satellite = base.Satellite(prn, azimuth, elevation, snr)
            beaconInformation.seenBeacons.add(satellite)

    def _fixGSA(self):
        """
        Extracts the information regarding which satellites were used in
        obtaining the GPS fix from a GSA sentence.

        Precondition: A GSA sentence was fired. Postcondition: The current
        sentence data (C{self._sentenceData} will contain a set of the
        currently used PRNs (under the key C{_usedPRNs}.
        """
        self._sentenceData['_usedPRNs'] = set()
        for key in ('usedSatellitePRN_%d' % (x,) for x in range(12)):
            prn = getattr(self.currentSentence, key, None)
            if prn is not None:
                self._sentenceData['_usedPRNs'].add(int(prn))
    _SPECIFIC_SENTENCE_FIXES = {'GPGSV': _fixGSV, 'GPGSA': _fixGSA}

    def _sentenceSpecificFix(self):
        """
        Executes a fix for a specific type of sentence.
        """
        fixer = self._SPECIFIC_SENTENCE_FIXES.get(self.currentSentence.type)
        if fixer is not None:
            fixer(self)
    _FIXERS = {'type': lambda self: self._sentenceSpecificFix(), 'timestamp': lambda self: self._fixTimestamp(), 'datestamp': lambda self: self._fixDatestamp(), 'latitudeFloat': lambda self: self._fixCoordinateFloat(Angles.LATITUDE), 'latitudeHemisphere': lambda self: self._fixHemisphereSign(Angles.LATITUDE, 'latitude'), 'longitudeFloat': lambda self: self._fixCoordinateFloat(Angles.LONGITUDE), 'longitudeHemisphere': lambda self: self._fixHemisphereSign(Angles.LONGITUDE, 'longitude'), 'altitude': lambda self: self._convert('altitude', converter=lambda strRepr: base.Altitude(float(strRepr))), 'altitudeUnits': lambda self: self._fixUnits(unitKey='altitudeUnits'), 'heightOfGeoidAboveWGS84': lambda self: self._convert('heightOfGeoidAboveWGS84', converter=lambda strRepr: base.Altitude(float(strRepr))), 'heightOfGeoidAboveWGS84Units': lambda self: self._fixUnits(unitKey='heightOfGeoidAboveWGS84Units'), 'trueHeading': lambda self: self._statefulUpdate('trueHeading'), 'magneticVariation': lambda self: self._statefulUpdate('magneticVariation'), 'magneticVariationDirection': lambda self: self._fixHemisphereSign(Angles.VARIATION, 'heading'), 'speedInKnots': lambda self: self._fixUnits(valueKey='speed', sourceKey='speedInKnots', unit='N'), 'positionDilutionOfPrecision': lambda self: self._statefulUpdate('positionDilutionOfPrecision'), 'horizontalDilutionOfPrecision': lambda self: self._statefulUpdate('horizontalDilutionOfPrecision'), 'verticalDilutionOfPrecision': lambda self: self._statefulUpdate('verticalDilutionOfPrecision')}

    def clear(self):
        """
        Resets this adapter.

        This will empty the adapter state and the current sentence data.
        """
        self._state = {}
        self._sentenceData = {}

    def sentenceReceived(self, sentence):
        """
        Called when a sentence is received.

        Will clean the received NMEAProtocol sentence up, and then update the
        adapter's state, followed by firing the callbacks.

        If the received sentence was invalid, the state will be cleared.

        @param sentence: The sentence that is received.
        @type sentence: L{NMEASentence}
        """
        self.currentSentence = sentence
        self._sentenceData = {}
        try:
            self._validateCurrentSentence()
            self._cleanCurrentSentence()
        except base.InvalidSentence:
            self.clear()
        self._updateState()
        self._fireSentenceCallbacks()

    def _validateCurrentSentence(self):
        """
        Tests if a sentence contains a valid fix.
        """
        if self.currentSentence.fixQuality is GPGGAFixQualities.INVALID_FIX or self.currentSentence.dataMode is GPGLLGPRMCFixQualities.VOID or self.currentSentence.fixType is GPGSAFixTypes.GSA_NO_FIX:
            raise base.InvalidSentence('bad sentence')

    def _cleanCurrentSentence(self):
        """
        Cleans the current sentence.
        """
        for key in sorted(self.currentSentence.presentAttributes):
            fixer = self._FIXERS.get(key, None)
            if fixer is not None:
                fixer(self)

    def _updateState(self):
        """
        Updates the current state with the new information from the sentence.
        """
        self._updateBeaconInformation()
        self._combineDateAndTime()
        self._state.update(self._sentenceData)

    def _updateBeaconInformation(self):
        """
        Updates existing beacon information state with new data.
        """
        new = self._sentenceData.get('_partialBeaconInformation')
        if new is None:
            return
        self._updateUsedBeacons(new)
        self._mergeBeaconInformation(new)
        if self.currentSentence._isLastGSVSentence():
            if not self.currentSentence._isFirstGSVSentence():
                del self._state['_partialBeaconInformation']
            bi = self._sentenceData.pop('_partialBeaconInformation')
            self._sentenceData['beaconInformation'] = bi

    def _updateUsedBeacons(self, beaconInformation):
        """
        Searches the adapter state and sentence data for information about
        which beacons where used, then adds it to the provided beacon
        information object.

        If no new beacon usage information is available, does nothing.

        @param beaconInformation: The beacon information object that beacon
            usage information will be added to (if necessary).
        @type beaconInformation: L{twisted.positioning.base.BeaconInformation}
        """
        for source in [self._state, self._sentenceData]:
            usedPRNs = source.get('_usedPRNs')
            if usedPRNs is not None:
                break
        else:
            return
        for beacon in beaconInformation.seenBeacons:
            if beacon.identifier in usedPRNs:
                beaconInformation.usedBeacons.add(beacon)

    def _mergeBeaconInformation(self, newBeaconInformation):
        """
        Merges beacon information in the adapter state (if it exists) into
        the provided beacon information. Specifically, this merges used and
        seen beacons.

        If the adapter state has no beacon information, does nothing.

        @param newBeaconInformation: The beacon information object that beacon
            information will be merged into (if necessary).
        @type newBeaconInformation: L{twisted.positioning.base.BeaconInformation}
        """
        old = self._state.get('_partialBeaconInformation')
        if old is None:
            return
        for attr in ['seenBeacons', 'usedBeacons']:
            getattr(newBeaconInformation, attr).update(getattr(old, attr))

    def _combineDateAndTime(self):
        """
        Combines a C{datetime.date} object and a C{datetime.time} object,
        collected from one or more NMEA sentences, into a single
        C{datetime.datetime} object suitable for sending to the
        L{IPositioningReceiver}.
        """
        if not any((k in self._sentenceData for k in ['_date', '_time'])):
            return
        date, time = (self._sentenceData.get(key) or self._state.get(key) for key in ('_date', '_time'))
        if date is None or time is None:
            return
        dt = datetime.datetime.combine(date, time)
        self._sentenceData['time'] = dt

    def _fireSentenceCallbacks(self):
        """
        Fires sentence callbacks for the current sentence.

        A callback will only fire if all of the keys it requires are present
        in the current state and at least one such field was altered in the
        current sentence.

        The callbacks will only be fired with data from L{_state}.
        """
        iface = ipositioning.IPositioningReceiver
        for name, method in iface.namesAndDescriptions():
            callback = getattr(self._receiver, name)
            kwargs = {}
            atLeastOnePresentInSentence = False
            try:
                for field in method.positional:
                    if field in self._sentenceData:
                        atLeastOnePresentInSentence = True
                    kwargs[field] = self._state[field]
            except KeyError:
                continue
            if atLeastOnePresentInSentence:
                callback(**kwargs)