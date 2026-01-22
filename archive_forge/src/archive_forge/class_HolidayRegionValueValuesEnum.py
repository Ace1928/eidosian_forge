from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HolidayRegionValueValuesEnum(_messages.Enum):
    """The geographical region based on which the holidays are considered in
    time series modeling. If a valid value is specified, then holiday effects
    modeling is enabled.

    Values:
      HOLIDAY_REGION_UNSPECIFIED: Holiday region unspecified.
      GLOBAL: Global.
      NA: North America.
      JAPAC: Japan and Asia Pacific: Korea, Greater China, India, Australia,
        and New Zealand.
      EMEA: Europe, the Middle East and Africa.
      LAC: Latin America and the Caribbean.
      AE: United Arab Emirates
      AR: Argentina
      AT: Austria
      AU: Australia
      BE: Belgium
      BR: Brazil
      CA: Canada
      CH: Switzerland
      CL: Chile
      CN: China
      CO: Colombia
      CS: Czechoslovakia
      CZ: Czech Republic
      DE: Germany
      DK: Denmark
      DZ: Algeria
      EC: Ecuador
      EE: Estonia
      EG: Egypt
      ES: Spain
      FI: Finland
      FR: France
      GB: Great Britain (United Kingdom)
      GR: Greece
      HK: Hong Kong
      HU: Hungary
      ID: Indonesia
      IE: Ireland
      IL: Israel
      IN: India
      IR: Iran
      IT: Italy
      JP: Japan
      KR: Korea (South)
      LV: Latvia
      MA: Morocco
      MX: Mexico
      MY: Malaysia
      NG: Nigeria
      NL: Netherlands
      NO: Norway
      NZ: New Zealand
      PE: Peru
      PH: Philippines
      PK: Pakistan
      PL: Poland
      PT: Portugal
      RO: Romania
      RS: Serbia
      RU: Russian Federation
      SA: Saudi Arabia
      SE: Sweden
      SG: Singapore
      SI: Slovenia
      SK: Slovakia
      TH: Thailand
      TR: Turkey
      TW: Taiwan
      UA: Ukraine
      US: United States
      VE: Venezuela
      VN: Viet Nam
      ZA: South Africa
    """
    HOLIDAY_REGION_UNSPECIFIED = 0
    GLOBAL = 1
    NA = 2
    JAPAC = 3
    EMEA = 4
    LAC = 5
    AE = 6
    AR = 7
    AT = 8
    AU = 9
    BE = 10
    BR = 11
    CA = 12
    CH = 13
    CL = 14
    CN = 15
    CO = 16
    CS = 17
    CZ = 18
    DE = 19
    DK = 20
    DZ = 21
    EC = 22
    EE = 23
    EG = 24
    ES = 25
    FI = 26
    FR = 27
    GB = 28
    GR = 29
    HK = 30
    HU = 31
    ID = 32
    IE = 33
    IL = 34
    IN = 35
    IR = 36
    IT = 37
    JP = 38
    KR = 39
    LV = 40
    MA = 41
    MX = 42
    MY = 43
    NG = 44
    NL = 45
    NO = 46
    NZ = 47
    PE = 48
    PH = 49
    PK = 50
    PL = 51
    PT = 52
    RO = 53
    RS = 54
    RU = 55
    SA = 56
    SE = 57
    SG = 58
    SI = 59
    SK = 60
    TH = 61
    TR = 62
    TW = 63
    UA = 64
    US = 65
    VE = 66
    VN = 67
    ZA = 68