 skip forward in a file until a given string is found

     **Arguments**

       - inFile: a file object (or anything supporting a _readline()_ method)

       - matchStr: the string to search for

       - fullMatch: if nonzero, _matchStr_ must match the entire line

     **Returns**

       the matching line

     **Notes:**

       - if _matchStr_ is not found in the file, a NoMatchFound exception
         will be raised

  